#include "ImageProcessingGameModeBase.h"


#include "ImageUtils.h"



#include "PreOpenCVHeaders.h"
#include "gdal.h"
#include "gdal_priv.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml.hpp"
#include "PostOpenCVHeaders.h"


const FString AImageProcessingGameModeBase::IMAGES_DIR_NAME = TEXT("Images");

AImageProcessingGameModeBase::AImageProcessingGameModeBase()
	: ImagesDirectory(FPaths::Combine(FPaths::ProjectDir(), IMAGES_DIR_NAME)), bHasImageLoaded(false)
{
}

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
	
	FString Extension = FPaths::GetExtension(ImagePath);
	if (Extension.Equals(TEXT("jp2")))
	{
		GDALDataset* loadedDataset = (GDALDataset*)GDALOpen(TCHAR_TO_ANSI(*ImagePath), GA_ReadOnly);
		if (loadedDataset == nullptr)
			return nullptr;

		int width = loadedDataset->GetRasterXSize();
		int height = loadedDataset->GetRasterYSize();
		int numBands = loadedDataset->GetRasterCount();
		
		uint16* data = new uint16[width * height * numBands];

        loadedDataset->RasterIO(GF_Read, 0,0, width, height, data, width, height, GDT_UInt16, numBands, NULL, 0,0,0);
		cv::Mat image(width, height, CV_16UC(numBands), data);
        cv::normalize(image,image, 0, 255,cv::NORM_MINMAX,CV_8U);
		cv::Ptr<cv::CLAHE> Clahe = cv::createCLAHE();

		Clahe->apply(image, image);
		Clahe.release();

		Image = createImageFromMat(image);
	} else
	{
		bool bLoadSuccess = FImageUtils::LoadImage(*ImagePath, Image);
		if (!bLoadSuccess)
			return nullptr;
	}


	UTexture2D* Texture = ConvertToTexture2D(Image);
	if (!IsValid(Texture))
		return nullptr;

	ImagesHistory.Empty();
	ImagesHistory.Add(Image);


	return Texture;
}

UTexture2D* AImageProcessingGameModeBase::UndoAction()
{
	if ( ImagesHistory.IsEmpty())
		return nullptr;
	if (ImagesHistory.Num() == 1)
	{
		ImagesHistory.Empty();
		return nullptr;
	}
	ImagesHistory.Pop();

	return ConvertToTexture2D(ImagesHistory.Top());	
}

bool AImageProcessingGameModeBase::SaveImage(const FString& NewImagePath)
{
	
	if ( ImagesHistory.IsEmpty())
		return nullptr;
	
	return FImageUtils::SaveImageAutoFormat(*NewImagePath, ImagesHistory.Pop());
}

UTexture2D* AImageProcessingGameModeBase::ApplyThreshold(double LimitValue)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::threshold(SourceMat, DestMat, LimitValue, 255, cv::THRESH_BINARY);

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

	return Texture;
}

UTexture2D* AImageProcessingGameModeBase::ApplyAverageBlur(FIntPoint KernelSize)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = CreateMatFromImage(ImagesHistory.Top());
	cv::Mat DestMat;
	cv::blur(SourceMat, DestMat, {KernelSize.X, KernelSize.Y}, cv::Point(-1, -1));

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
	cv::medianBlur(SourceMat, DestMat, Aperture);

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

	cv::filter2D(SourceMat, DestMat, -1, Kernel, cv::Point(-1, -1));

	// Ignore work done in alpha channel
	if (DestMat.type() == CV_8UC4)
	{
		int index[] = {3, 3};
		cv::mixChannels(&SourceMat, 1, &DestMat, 1, index, 1);
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
	cv::blur(SourceMat, BlurMat, {KernelSize, KernelSize}, cv::Point(-1, -1));
	cv::addWeighted(SourceMat, BoostValue + 1, BlurMat, -BoostValue, 0, DestMat);

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);

	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::ApplyRoberts(FilterDirection Direction, bool bUseZeroCross)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat, VerticalMask, HorizontalMask;

	cv::filter2D(SourceMat, HorizontalMask, CV_16S, X_ROBERTS_KERNEL);
	cv::filter2D(SourceMat, VerticalMask, CV_16S, Y_ROBERTS_KERNEL);

	switch (Direction)
	{
	case FilterDirection::Horizontal:
		DestMat = HorizontalMask;
		break;
	case FilterDirection::Vertical:
		DestMat = VerticalMask;
		break;
	default:
		DestMat = VerticalMask + HorizontalMask;
		break;
	}
	if (bUseZeroCross)
		DestMat = ApplyZeroCross(DestMat);
	else
		cv::convertScaleAbs(DestMat, DestMat);

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::ApplyPrewitt(FilterDirection Direction, bool bUseZeroCross)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat, VerticalMask, HorizontalMask;

	cv::filter2D(SourceMat, HorizontalMask, CV_16S, X_PREWITT_KERNEL);
	cv::filter2D(SourceMat, VerticalMask, CV_16S, Y_PREWITT_KERNEL);

	switch (Direction)
	{
	case FilterDirection::Horizontal:
		DestMat = HorizontalMask;
		break;
	case FilterDirection::Vertical:
		DestMat = VerticalMask;
		break;
	default:
		DestMat = VerticalMask + HorizontalMask;
		break;
	}
	if (bUseZeroCross)
		DestMat = ApplyZeroCross(DestMat);
	else
		cv::convertScaleAbs(DestMat, DestMat);

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::ApplySobel(FilterDirection Direction, int32 Size, bool bUseZeroCross)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat;
	cv::Mat HorizontalMat, VerticalMat;

	cv::Sobel(SourceMat, HorizontalMat, CV_16S, 1, 0, Size);
	cv::Sobel(SourceMat, VerticalMat, CV_16S, 1, 0, Size);

	switch (Direction)
	{
	case FilterDirection::Horizontal:
		DestMat = HorizontalMat;
		break;
	case FilterDirection::Vertical:
		DestMat = VerticalMat;
		break;
	default:
		DestMat = VerticalMat + HorizontalMat;
		break;
	}
	if (bUseZeroCross)
		DestMat = ApplyZeroCross(DestMat);
	else
		cv::convertScaleAbs(DestMat, DestMat);

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::ApplyCanny(int32 Min, int32 Max, int32 SobelAperture)
{
	if (ImagesHistory.IsEmpty()) return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat DestMat;

	cv::Canny(SourceMat, DestMat, Min, Max, SobelAperture);
	cv::convertScaleAbs(DestMat, DestMat);

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::ApplyLaplacianOfGauss(int32 KernelSize, bool bApplyZeroCross)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	cv::Mat Gaus, Laplacian;
	cv::GaussianBlur(SourceMat, Gaus, {KernelSize, KernelSize}, 0);
	cv::Laplacian(Gaus, Laplacian,CV_16S, KernelSize);

	cv::Mat Final;
	if (bApplyZeroCross)
		Final = ApplyZeroCross(Laplacian);
	else
		cv::convertScaleAbs(Laplacian,Final);

	FImage NewImage = createImageFromMat(Final);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

cv::Mat AImageProcessingGameModeBase::ApplyZeroCross(const cv::Mat& Image)
{
	cv::Mat DestMat(Image.size(), CV_8U, cv::Scalar(0));

	for (int i = 0; i < Image.rows - 1; i++)
	{
		for (int j = 0; j < Image.cols - 1; j++)
		{
			if (Image.at<short>(i, j) > 0)
			{
				if (Image.at<short>(i + 1, j) < 0 || Image.at<short>(i + 1, j + 1) < 0 || Image.at<short>(i, j + 1) < 0)
				{
					DestMat.at<uint8>(i, j) = 255;
				}
			}
			else if (Image.at<short>(i, j) < 0)
			{
				if (Image.at<short>(i + 1, j) > 0 || Image.at<short>(i + 1, j + 1) > 0 || Image.at<short>(i, j + 1) > 0)
				{
					DestMat.at<uint8>(i, j) = 255;
				}
			}
		}
	}

	return DestMat;
}

UTexture2D* AImageProcessingGameModeBase::ApplyWatershed(int32 KernelSize, float ForeGroundMultiplier)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat Source = CreateMatFromImage(ImagesHistory.Top());
	if (Source.channels() == 1)
	{
		cv::cvtColor(Source, Source, cv::COLOR_GRAY2BGR);
	}
	else
	{
		cv::cvtColor(Source, Source, cv::COLOR_BGRA2BGR);
	}

	cv::Mat Gray;
	cv::cvtColor(Source, Gray, cv::COLOR_BGR2GRAY);

	// Aplly OTSU Threshold
	cv::Mat Thresholded;
	cv::threshold(Gray, Thresholded, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::bitwise_not(Thresholded, Thresholded);

	cv::Mat kernel = cv::Mat::ones(KernelSize, KernelSize,CV_8U);
	cv::Mat Opened;
	cv::morphologyEx(Thresholded, Opened, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 3);

	cv::Mat SureBG;
	cv::dilate(Opened, SureBG, kernel, cv::Point(-1, -1), 3);

	cv::Mat Distance, SureFG;
	cv::distanceTransform(Opened, Distance, cv::DIST_L2, 5);
	normalize(Distance, Distance, 0, 1.0, cv::NORM_MINMAX);
	double min, max;
	cv::minMaxLoc(Distance, &min, &max);
	cv::threshold(Distance, SureFG, ForeGroundMultiplier * max, 255, cv::THRESH_BINARY);
	cv::convertScaleAbs(SureFG, SureFG);

	cv::Mat Unknown;
	cv::subtract(SureBG, SureFG, Unknown);

	cv::Mat Markers(SureFG.size(), CV_32SC1);
	cv::connectedComponents(SureFG, Markers);
	Markers += 1;

	for (int i = 0; i < Markers.rows; i++)
	{
		for (int j = 0; j < Markers.cols; j++)
		{
			if (Unknown.at<uchar>(i, j) == 255)
			{
				Markers.at<int>(i, j) = 0;
			}
		}
	}

	cv::watershed(Source, Markers);
	for (int i = 0; i < Markers.rows; i++)
	{
		for (int j = 0; j < Markers.cols; j++)
		{
			if (Markers.at<int>(i, j) == -1)
			{
				Source.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
			}
		}
	}
	cv::cvtColor(Source, Source, cv::COLOR_BGR2BGRA);

	FImage NewImage = createImageFromMat(Source);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::ObjectCount(int32 Threshold, int32 MinObjectArea, int32& OutCount)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	const FImageView& Image = ImagesHistory.Top();
	cv::Mat Source = CreateMatFromImage(Image);
	cv::Mat ConvSource;
	if (Source.channels() == 1)
	{
		cv::cvtColor(Source, ConvSource, cv::COLOR_GRAY2BGR);
	}
	else
	{
		cv::cvtColor(Source, ConvSource, cv::COLOR_BGRA2BGR);
	}

	cv::Mat Filtered;
	cv::pyrMeanShiftFiltering(ConvSource, Filtered, 20, 40);

	cv::Mat GraySource;
	cvtColor(Filtered, GraySource, cv::COLOR_BGR2GRAY);

	cv::Mat Thresholded;
	cv::threshold(GraySource, Thresholded, Threshold, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> Contours;
	std::vector<cv::Vec4i> Hierarchy;
	findContours(Thresholded, Contours, Hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat Drawing;
	cv::cvtColor(Thresholded, Drawing, cv::COLOR_GRAY2BGRA);

	OutCount = 0;
	for (size_t i = 0; i < Contours.size(); i++)
	{
		double Area = cv::contourArea(Contours[i]);
		if (Area > MinObjectArea)
		{
			OutCount++;
			cv::Scalar Color = cv::Scalar(FMath::RandRange(0, 255), FMath::RandRange(0, 255), FMath::RandRange(0, 255),
			                              255);
			drawContours(Drawing, Contours, (int)i, Color, 5, cv::LINE_8, Hierarchy, 0);
		}
	}
	
	std::stringstream ss;
	ss << "Foram encontrados " << OutCount << " Objetos";
	cv::putText(Drawing,ss.str(),cv::Point(10,30),cv::FONT_HERSHEY_SIMPLEX, 0.5,{0,255,0},1,cv::LINE_AA);

	FImage NewImage = createImageFromMat(Drawing);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
}

UTexture2D* AImageProcessingGameModeBase::NormalizedDifferenceFromImage(const FString& ImagePath)
{
	if (ImagesHistory.IsEmpty())
		return nullptr;
	
	UTexture2D* BImage = LoadImage(ImagePath);
	if (BImage == nullptr)
		return nullptr;

	cv::Mat ImageB = CreateMatFromImage(ImagesHistory.Pop());
	cv::Mat ImageA = CreateMatFromImage(ImagesHistory.Top());
	
	cv::Mat DestMat = ImageA - ImageB / (ImageA + ImageB);
	
	FImage Image = createImageFromMat(DestMat);
	ImagesHistory.Add(Image);
	return ConvertToTexture2D(Image);
}

void AImageProcessingGameModeBase::GenerateHistogram(TArray<int>& GrayValues, TArray<int>& BlueValues, TArray<int>& RedValues, TArray<int>& GreenValues)
{
	if (ImagesHistory.IsEmpty())
		return;

	cv::Mat Source = CreateMatFromImage(ImagesHistory.Top());
	if (Source.channels() == 4)
		cv::cvtColor(Source, Source, cv::COLOR_BGRA2BGR);

	int32 HistogramSize = 255;
	float Range[2] = {0, 256};
	const float* HistogramRange = Range;

	if (Source.channels() == 3)
	{
		cv::Mat RedChannel, GreenChannel, BlueChannel;
		cv::Mat SourceChannels[3];

		cv::split(Source, SourceChannels);
		cv::calcHist(&SourceChannels[0], 1, 0, cv::Mat(), BlueChannel, 1, &HistogramSize, &HistogramRange);
		cv::calcHist(&SourceChannels[1], 1, 0, cv::Mat(), GreenChannel, 1, &HistogramSize, &HistogramRange);
		cv::calcHist(&SourceChannels[2], 1, 0, cv::Mat(), RedChannel, 1, &HistogramSize, &HistogramRange);

		for (int i = 0; i < HistogramSize; i++)
		{
			GreenValues.Add(cvRound(GreenChannel.at<float>(i)));
			RedValues.Add(cvRound(RedChannel.at<float>(i)));
			BlueValues.Add(cvRound(BlueChannel.at<float>(i)));
		}
	}
	else
	{
		cv::Mat GrayChannel;
		cv::calcHist(&Source, 1, 0, cv::Mat(), GrayChannel, 1, &HistogramSize, &HistogramRange);
		
		for (int i = 0; i < HistogramSize; i++)
		{
			GrayValues.Add(cvRound(GrayChannel.at<float>(i)));
		}
	}
}

UTexture2D* AImageProcessingGameModeBase::ApplyAdaptiveHistogramEqualization()
{
	if (ImagesHistory.IsEmpty())
		return nullptr;

	cv::Mat SourceMat = ConvertToGrayScale(CreateMatFromImage(ImagesHistory.Top()));
	
	cv::Ptr<cv::CLAHE> Clahe = cv::createCLAHE();

	cv::Mat DestMat;
	Clahe->apply(SourceMat, DestMat);
	Clahe.release();
	
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
		int64 RandomCol = FMath::RandRange(0, SourceMat.cols - 1 );
		int64 RandomRow = FMath::RandRange(0, SourceMat.rows - 1 );
		uint8 NoiseValue = FMath::RandBool() ? 255 : 0;

		FMemory::Memset(DestMat.ptr(RandomRow, RandomCol), NoiseValue, DestMat.channels());
	}

	FImage NewImage = createImageFromMat(DestMat);
	ImagesHistory.Add(NewImage);
	return ConvertToTexture2D(NewImage);
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
	if (Matrix.channels() == 3)
	{
		cv::cvtColor(Matrix, TempMat, cv::COLOR_BGR2GRAY);
	}
	else if (Matrix.channels() == 4)
	{
		cv::cvtColor(Matrix, TempMat, cv::COLOR_BGRA2GRAY);
	}
	else
	{
		TempMat = Matrix.clone();
	}
	return TempMat;
}
