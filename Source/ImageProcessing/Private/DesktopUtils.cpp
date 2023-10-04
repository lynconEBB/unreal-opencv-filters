#include "DesktopUtils.h"

#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "AssetRegistry/AssetRegistryModule.h"

void UDesktopUtils::OpenFileDialog(const FString& DialogTitle, const FString& DefaultPath, const FString& FileTypes,
                                   TArray<FString>& OutFileNames)
{
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();

	if (DesktopPlatform == nullptr )
		return;
	
	const void* WindowHandle = FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
	DesktopPlatform->OpenFileDialog(WindowHandle, DialogTitle, DefaultPath,TEXT(""), FileTypes, EFileDialogFlags::None, OutFileNames);
}

bool UDesktopUtils::MakeDirectory(const FString& a)
{
	FPaths::ProjectDir();
	IFileManager& FileManager = IFileManager::Get();
	return FileManager.MakeDirectory(*a);
}
