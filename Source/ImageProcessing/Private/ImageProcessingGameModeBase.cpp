#include "ImageProcessingGameModeBase.h"


#include "ImageUtils.h"

#include "PreOpenCVHeaders.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml.hpp"
#include "PostOpenCVHeaders.h"

static int8 PrewittX[] = {1,1,1,0,0,0,-1,-1,-1};
static int8 PrewittY[] = {-1,0,1,-1,0,1,-1,0,1};
static const cv::Mat X_PREWITT_KERNEL(cv::Size(3,3),CV_8S,PrewittX);
static const cv::Mat Y_PREWITT_KERNEL(3,3,CV_8S, PrewittY);
static int8 RobertsX[] = {-1,0,0,1};
static int8 RobertsY[] = {0,-1,1,0};
static const cv::Mat X_ROBERTS_KERNEL(2,2,CV_8S, RobertsX);
static const cv::Mat Y_ROBERTS_KERNEL(2,2,CV_8S, RobertsY);

const FString AImageProcessingGameModeBase::IMAGES_DIR_NAME = TEXT("Images");

AImageProcessingGameModeBase::AImageProcessingGameModeBase()
	: ImagesDirectory(FPaths::Combine(FPaths::ProjectDir(), IMAGES_DIR_NAME)), bHasImageLoaded(false)
{ }

void AImageProcessingGameModeBase::BeginPlay()
{
	Super::BeginPlay();

	IFileManager& FileManager = IFileManager::Get();
	FileManager.DeleteDirectory(*ImagesDirectory);
	FileManager.MakeDirectory(*ImagesDirectory);
}


UTexture2D* AImageProcessingGameModeBase::LoadImage(const FString& ImagePath)
{
	FImage Image;
	
	bool bLoadSuccess = FImageUtils::LoadImage(*ImagePath, Image);
	if (!bLoadSuccess)
		return nullptr;

	UTexture2D* Texture = ConvertToTexture2D(Image);
	if(!IsValid(Texture))
		return nullptr;

	ImagesHistory.Empty();
	ImagesHistory.Add(Image);

	
	return Texture;
}

UTexture2D* AImageProcessingGameModeBase::ApplyThreshold(double LimitValue, ThresholdType Type)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::threshold(SourceMat,DestMat ,LimitValue,255, Type);

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ToGrayScale()
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat = ConvertToGrayScale(SourceMat);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);

	UTexture2D* Texture = ConvertToTexture2D(NewImage);
	
	return  Texture;
}

UTexture2D* AImageProcessingGameModeBase::ApplyAverageBlur(FIntPoint KernelSize)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::blur(SourceMat, DestMat, {KernelSize.X, KernelSize.Y},cv::Point(-1,-1));
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplyMedianBlur(int32 Aperture)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;
	
	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::medianBlur(SourceMat, DestMat,Aperture);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplyHighPass(int32 KernelSize)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;
	
	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::Mat Kernel(KernelSize, KernelSize, CV_32F, -1);
	
	int32 MiddleIndex = UE4::SSE::FloorToInt32(KernelSize / 2.f);
	Kernel.at<float>(MiddleIndex, MiddleIndex) = (KernelSize * KernelSize) - 1;
	Kernel /= (KernelSize * KernelSize);

	cv::filter2D(SourceMat, DestMat, -1, Kernel,cv::Point(-1,-1));

	// Ignore work done in alpha channel
	if (DestMat.type() == CV_8UC4)
	{
		int index[] = {3,3};
		cv::mixChannels(&SourceMat, 1, &DestMat, 1,index,1);
	}
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplyHighBoost(int32 KernelSize, int32 BoostValue)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;
	
	cv::Mat BlurMat;
	cv::Mat DestMat;
	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::blur(SourceMat, BlurMat, {KernelSize, KernelSize},cv::Point(-1,-1));
	cv::addWeighted(SourceMat, BoostValue + 1, BlurMat, -BoostValue, 0, DestMat);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplyRoberts(FilterDirection Direction)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat, VerticalMask, HorizontalMask;

	cv::filter2D(SourceMat, HorizontalMask, -1, X_ROBERTS_KERNEL);
	cv::filter2D(SourceMat, VerticalMask, -1, Y_ROBERTS_KERNEL);

	switch (Direction) {
	case FilterDirection::Horizontal:
		DestMat =  HorizontalMask;
		break;
	case FilterDirection::Vertical:
		DestMat = VerticalMask;
		break;
	default:
		DestMat = VerticalMask + HorizontalMask;
		break;
	}
	cv::convertScaleAbs(DestMat, DestMat);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplyPrewitt(FilterDirection Direction)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat, VerticalMask, HorizontalMask;
	
	cv::filter2D(SourceMat, HorizontalMask, -1, X_PREWITT_KERNEL);
	cv::filter2D(SourceMat, VerticalMask, -1, Y_PREWITT_KERNEL);

	switch (Direction) {
	case FilterDirection::Horizontal:
		DestMat =  HorizontalMask;
		break;
	case FilterDirection::Vertical:
		DestMat = VerticalMask;
		break;
	default:
		DestMat = VerticalMask + HorizontalMask;
		break;
	}
	cv::convertScaleAbs(DestMat, DestMat);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplySobel(FilterDirection Direction, int32 Size)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat;
	cv::Mat HorizontalMat, VerticalMat;

	cv::Sobel(SourceMat, HorizontalMat, CV_16U, 1, 0, Size);
	cv::Sobel(SourceMat, VerticalMat, CV_16U, 1, 0, Size);
	
	switch (Direction) {
	case FilterDirection::Horizontal:
		DestMat =  HorizontalMat;
		break;
	case FilterDirection::Vertical:
		DestMat = VerticalMat;
		break;
	default:
		DestMat = VerticalMat + HorizontalMat;
		break;
	}
	cv::convertScaleAbs(DestMat, DestMat);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::ApplyLaplacianOfGauss(int32 KernelSize)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::GaussianBlur(SourceMat, DestMat,{KernelSize,KernelSize},0);
	cv::Laplacian(DestMat, DestMat,-1,KernelSize);


	cv::Mat Final = ApplyZeroCross(DestMat);
	
	FImage NewImage = createImageFromMat(Final);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage); 
}


cv::Mat AImageProcessingGameModeBase::ApplyZeroCross(const cv::Mat& Image)
{
	cv::Mat DestMat(Image.size(), CV_8U, cv::Scalar(0));

    for (int i = 0; i < Image.rows - 1; i++) {
        for (int j = 0; j < Image.cols - 1; j++) {
            if (Image.at<short>(i, j) > 0) {
                if (Image.at<short>(i + 1, j) < 0 || Image.at<short>(i + 1, j + 1) < 0 || Image.at<short>(i, j + 1) < 0) {
                    DestMat.at<uchar>(i, j) = 1;
                }
            } else if (Image.at<short>(i, j) < 0) {
                if (Image.at<short>(i + 1, j) > 0 || Image.at<short>(i + 1, j + 1) > 0 || Image.at<short>(i, j + 1) > 0) {
                    DestMat.at<uchar>(i, j) = 1;
                }
            }
        }
    }

    return DestMat;
}

UTexture2D* AImageProcessingGameModeBase::ApplyCanny(int32 Min, int32 Max, int32 SobelAperture)
{
	if (ImagesHistory.IsEmpty()) return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat;

	cv::Canny(SourceMat,DestMat, Min, Max, SobelAperture);
	cv::convertScaleAbs(DestMat, DestMat);
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage); 
}

UTexture2D* AImageProcessingGameModeBase::AddSaltAndPepper(float NoiseProbability)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat = SourceMat.clone();
	
	int64 NoiseCount = NoiseProbability * SourceMat.cols * SourceMat.rows;
	for (int64 i = 0; i < NoiseCount; i++)
	{
		int64 RandomCol = FMath::RandRange(0, SourceMat.cols);
		int64 RandomRow = FMath::RandRange(0, SourceMat.rows);
		uint8 NoiseValue = FMath::RandBool() ? 255 : 0;
		
		FMemory::Memset(DestMat.ptr(RandomRow,RandomCol), NoiseValue, SourceMat.channels());
	}
	
	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	UTexture2D* Texture = ConvertToTexture2D(NewImage);

	return Texture; 
}

UTexture2D* AImageProcessingGameModeBase::ConvertToTexture2D(const FImageView& Image)
{
	UTexture2D* Texture = FImageUtils::CreateTexture2DFromImage(Image);
	if (!IsValid(Texture))
		return nullptr;

	if (Image.Format == ERawImageFormat::G8)
	{
		Texture->CompressionSettings = TC_Grayscale;
		Texture->UpdateResource();
	}
	
	return Texture;
}

cv::Mat AImageProcessingGameModeBase::CreateMatFromImage(const FImageView& Image)
{
	int CvType = Image.Format == ERawImageFormat::G8 ? CV_8U : CV_8UC4;
	return cv::Mat(Image.SizeY, Image.SizeX, CvType, Image.RawData);
}

FImage AImageProcessingGameModeBase::createImageFromMat(const cv::Mat& Matrix)
{
	ERawImageFormat::Type ImageType = Matrix.type() == CV_8U ? ERawImageFormat::G8 : ERawImageFormat::BGRA8; 
	FImage NewImage(Matrix.cols, Matrix.rows, ImageType);

	int Size = Matrix.cols * Matrix.rows * Matrix.channels();
	NewImage.RawData.Reserve(Size);
	FMemory::Memcpy(NewImage.RawData.GetData(), Matrix.data, Size);
	
	return NewImage;
}

cv::Mat AImageProcessingGameModeBase::ConvertToGrayScale(const cv::Mat& Matrix)
{
	cv::Mat TempMat;
    if (Matrix.channels() == 3) {
        cv::cvtColor(Matrix, TempMat, cv::COLOR_BGR2GRAY);
    } else if (Matrix.channels() == 4) {
        cv::cvtColor(Matrix, TempMat, cv::COLOR_BGRA2GRAY);
    }else {
        TempMat = Matrix.clone();
    }
    return TempMat;
}
