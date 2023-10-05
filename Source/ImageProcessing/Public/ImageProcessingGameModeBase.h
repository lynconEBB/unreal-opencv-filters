#pragma once

#include "PreOpenCVHeaders.h"
#include "opencv2/core/mat.hpp"
#include "PostOpenCVHeaders.h"

#include "CoreMinimal.h"
#include "IntVectorTypes.h"
#include "GameFramework/GameModeBase.h"
#include "ImageProcessingGameModeBase.generated.h"

UENUM(BlueprintType)
enum ThresholdType : uint8
{
	Binary,
	BinaryInverse,
	Truncate,
	Zero,
	ZeroInverse
};

UENUM(BlueprintType)
enum class FilterDirection : uint8
{
	Horizontal,
	Vertical,
	Both,
};


UCLASS(BlueprintType)
class IMAGEPROCESSING_API AImageProcessingGameModeBase : public AGameModeBase
{
	GENERATED_BODY()
	
public:
	AImageProcessingGameModeBase();

	virtual void BeginPlay() override;
	
	UFUNCTION(BlueprintCallable)
	UTexture2D* LoadImage(const FString& ImagePath);
	UTexture2D* UndoAction();
	
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyThreshold(double LimitValue, ThresholdType Type);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ToGrayScale();
	UFUNCTION(BlueprintCallable)
	UTexture2D* AddSaltAndPepper(float NoiseProbability);

	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyAverageBlur(FIntPoint KernelSize);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyMedianBlur(int32 Aperture);
	
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyHighPass(int32 KernelSize);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyHighBoost(int32 KernelSize, int32 BoostValue);

	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyRoberts(FilterDirection Direction, bool bUseZeroCross);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyPrewitt(FilterDirection Direction, bool bUseZeroCross);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplySobel(FilterDirection Direction, int32 Size, bool bUseZeroCross);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyCanny(int32 Min, int32 Max, int32 SobelAperture);
	
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyLaplacianOfGauss(int32 KernelSize);
	cv::Mat ApplyZeroCross(const cv::Mat& SourceImage);

	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyWatershed();
	UFUNCTION(BlueprintCallable)
	UTexture2D* ObjectCount(int32 Threshold, int32 MinObjectArea, int32& OutCount);

	UFUNCTION(BlueprintCallable)
	UTexture2D* GenerateHistogram();
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyAdaptiveHistogramEqualization();
private:
	UTexture2D* ConvertToTexture2D(const FImageView& Image);
	cv::Mat	CreateMatFromImage(const FImageView& Image);
	FImage createImageFromMat(const cv::Mat& Matrix);
	cv::Mat ConvertToGrayScale(const cv::Mat& Matrix);
	
private:
	static const FString IMAGES_DIR_NAME;
	FString ImagesDirectory;
	bool bHasImageLoaded;
	TArray<FImage> ImagesHistory;
	
	int8 PrewittX[9] = {1,1,1,0,0,0,-1,-1,-1};
	int8 PrewittY[9] = {-1,0,1,-1,0,1,-1,0,1};
	const cv::Mat X_PREWITT_KERNEL = cv::Mat(3,3,CV_8S,PrewittX);
	const cv::Mat Y_PREWITT_KERNEL = cv::Mat(3,3,CV_8S, PrewittY);

	int8 RobertsX[4] = {-1,0,0,1};
	int8 RobertsY[4] = {0,-1,1,0};
	const cv::Mat X_ROBERTS_KERNEL = cv::Mat(2,2,CV_8S, RobertsX);
	const cv::Mat Y_ROBERTS_KERNEL = cv::Mat(2,2,CV_8S, RobertsY);
};
