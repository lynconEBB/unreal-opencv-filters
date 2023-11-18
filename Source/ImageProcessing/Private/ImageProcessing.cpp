// Copyright Epic Games, Inc. All Rights Reserved.

#include "ImageProcessing.h"

#include "UnrealGDAL.h"
#include "Modules/ModuleManager.h"


void IImageProcessingModule::StartupModule()
{
	IModuleInterface::StartupModule();
	FUnrealGDALModule* UnrealGdalModule = FModuleManager::Get().LoadModulePtr<FUnrealGDALModule>("UnrealGDAL");
	UnrealGdalModule->InitGDAL();
}

void IImageProcessingModule::ShutdownModule()
{
	IModuleInterface::ShutdownModule();
}

IMPLEMENT_PRIMARY_GAME_MODULE( IImageProcessingModule, ImageProcessing, "ImageProcessing" );