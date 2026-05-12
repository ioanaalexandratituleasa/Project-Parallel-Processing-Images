Dependențe și instalare

1. OpenMP
Nu necesită instalare separată — vine inclus în MSVC (Visual Studio).
Project Properties → C/C++ → Language → OpenMP Support → Yes (/openmp)


2. MPI — Microsoft MPI (MS-MPI)
Descărcat de pe Microsoft official website.
Configurare în Visual Studio:
C/C++ → General → Additional Include Directories:
C:\Program Files (x86)\Microsoft SDKs\MPI\Include
Linker → General → Additional Library Directories:
C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64
Linker → Input → Additional Dependencies:
msmpi.lib


3. Intel TBB — instalat prin NuGet  
În Visual Studio:
Project Properties → C/C++ → Language → OpenMP Support → Yes (/openmp)
Click dreapta pe proiect → Manage NuGet Packages → Browse
Caută: oneTBB
Instalează: inteltbb.devel.win (2023.0.0.716)
Instalează: inteltbb.redist.win (2023.0.0.716


Cerințe sistem:
Windows 10/11 x64
Visual Studio 2019 sau 2022
Microsoft MPI instalat


Dataset:
Brain MRI Images for Brain Tumor Detection
Sursa: Kaggle (navoneel/brain-mri-images-for-brain-tumor-detection)
Imagini JPG greyscale, redimensionate la 1024×1024

