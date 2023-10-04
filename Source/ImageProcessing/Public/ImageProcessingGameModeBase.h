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
	
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyThreshold(double LimitValue, ThresholdType Type);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ToGrayScale();

	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyAverageBlur(FIntPoint KernelSize);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyMedianBlur(int32 Aperture);
	
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyHighPass(int32 KernelSize);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyHighBoost(int32 KernelSize, int32 BoostValue);

	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyRoberts(FilterDirection Direction);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyPrewitt(FilterDirection Direction);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplySobel(FilterDirection Direction, int32 Size);

	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyLaplacianOfGauss(int32 KernelSize);
	
	cv::Mat ApplyZeroCross(const cv::Mat& SourceImage);
	UFUNCTION(BlueprintCallable)
	UTexture2D* ApplyCanny(int32 Min, int32 Max, int32 SobelAperture);

	UFUNCTION(BlueprintCallable)
	UTexture2D* AddSaltAndPepper(float NoiseProbability);

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
};
