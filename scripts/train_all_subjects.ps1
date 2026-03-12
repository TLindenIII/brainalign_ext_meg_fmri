param (
    [string]$Modality = "eeg",
    [string]$Epochs = "",
    [switch]$Resume
)

$ModalityUpper = $Modality.ToUpper()
Write-Host "Starting sequential multi-subject training for THINGS-${ModalityUpper}..."

for ($i = 1; $i -le 10; $i++) {
    Write-Host "============================================================"
    Write-Host "Training Subject $i / 10 (${ModalityUpper})"
    Write-Host "============================================================"
    
    $SubId = "{0:D2}" -f $i
    $CheckpointPath = "checkpoints\${Modality}\${Modality}_brainalign_sub${SubId}_best.pt"
    
    if ((Test-Path $CheckpointPath) -and -not $Resume) {
        Write-Host "Checkpoint for Subject $i already exists. Skipping training..."
        Write-Host ""
        continue
    }
    
    $EpochArg = if ($Epochs) { "--epochs", $Epochs } else { @() }
    $ResumeArg = if ($Resume) { "--resume" } else { @() }
    
    & ".\venv\Scripts\python.exe" -m src.train --modality $Modality --subject $i @EpochArg @ResumeArg
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed for Subject $i. Exiting."
        exit $LASTEXITCODE
    }
    
    Write-Host "Finished training Subject $i."
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "All 10 subjects have been trained! Automatically generating the final averaged SOTA matrices..."
Write-Host "============================================================"

& ".\venv\Scripts\python.exe" -m src.evaluate_table
