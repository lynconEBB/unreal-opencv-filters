// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class ImageProcessing : ModuleRules
{
	public ImageProcessing(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		
	
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" ,"ImageCore" });
		PrivateDependencyModuleNames.AddRange(new string[] { "OpenCV","OpenCVHelper", "DesktopPlatform", "Slate", "SlateCore" });
	}
}
