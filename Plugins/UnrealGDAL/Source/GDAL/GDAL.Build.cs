
using System;
using System.IO;
using UnrealBuildTool;
using System.Collections.Generic;

public class GDAL : ModuleRules
{
	public GDAL(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		
		//Add include directory
		PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "include"));

		//add shared libs.
		var libs = Directory.GetFiles(Path.Combine(ModuleDirectory, "lib"), "*" + ".lib", SearchOption.AllDirectories);
		foreach (string lib in libs)
		{
			PublicAdditionalLibraries.Add(lib);
		}

		//add dlls
		var dlls = new List<string>(Directory.GetFiles(Path.Combine(ModuleDirectory, "bin"), "*" + ".dll"));
		string binaryStagingDir = Path.Combine("$(ProjectDir)", "Binaries", "Win64");

        foreach (string dll in dlls)
		{
			RuntimeDependencies.Add(Path.Combine(binaryStagingDir, Path.GetFileName(dll)), dll, StagedFileType.NonUFS);
		}

        PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
			}
		);
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"libcurl", //link against these modules so GDAL can use them.
				"SQLiteCore", //
				"LibTiff",
				"zlib",
			}
		);
	}
}
