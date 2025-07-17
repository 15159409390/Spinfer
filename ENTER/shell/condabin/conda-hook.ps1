$Env:CONDA_EXE = "/mnt/sevenT/spinfer/SpInfer_EuroSys25/ENTER/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/mnt/sevenT/spinfer/SpInfer_EuroSys25/ENTER"
$Env:_CONDA_EXE = "/mnt/sevenT/spinfer/SpInfer_EuroSys25/ENTER/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs