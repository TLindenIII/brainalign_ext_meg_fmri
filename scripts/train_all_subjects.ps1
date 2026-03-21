param (
    [string]$Modality = "eeg",
    [string]$Epochs = "",
    [switch]$Resume,
    [switch]$SharedOnly,
    [string]$SharedManifest = ""
)

$ModalityUpper = $Modality.ToUpper()
Write-Host "Starting sequential multi-subject training for THINGS-${ModalityUpper}..."

function Get-SubjectIds {
    param ([string]$Mode)

    switch ($Mode) {
        "eeg" {
            return Get-ChildItem "data\things-eeg2\preprocessed" -Directory -Filter "sub-*" |
                ForEach-Object { [int]($_.Name -replace "^sub-", "") } |
                Sort-Object
        }
        "meg" {
            return Get-ChildItem "data\things-meg-ds004212\derivatives\preprocessed" -File -Filter "preprocessed_P*-epo*.fif" |
                ForEach-Object {
                    if ($_.Name -match "preprocessed_P(\d+)-epo(?:-\d+)?\.fif") {
                        [int]$Matches[1]
                    }
                } |
                Sort-Object -Unique
        }
        "fmri" {
            return Get-ChildItem "data\things-fmri-ds004192\derivatives\ICA-betas" -Directory -Filter "sub-*" |
                ForEach-Object { [int]($_.Name -replace "^sub-", "") } |
                Sort-Object
        }
        default {
            throw "Unknown modality '$Mode'"
        }
    }
}

$SubjectIds = @(Get-SubjectIds $Modality)
if ($SubjectIds.Count -eq 0) {
    Write-Host "No local subjects found for modality '$Modality'."
    exit 1
}

$CheckpointSuffix = if ($SharedOnly) { "_shared" } else { "" }
$CheckpointStemExtra = if ($Modality -eq "meg") { "_attnpool" } else { "" }
$PythonExe = if (Test-Path ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} elseif (Test-Path ".\venv\Scripts\python.exe") {
    ".\venv\Scripts\python.exe"
} else {
    "python"
}

for ($idx = 0; $idx -lt $SubjectIds.Count; $idx++) {
    $i = $SubjectIds[$idx]
    Write-Host "============================================================"
    Write-Host "Training Subject $i / $($SubjectIds.Count) (${ModalityUpper})"
    Write-Host "============================================================"
    
    $SubId = "{0:D2}" -f $i
    $CheckpointPath = "checkpoints\${Modality}\${Modality}_brainalign_sub${SubId}${CheckpointStemExtra}${CheckpointSuffix}_best.pt"
    
    if ((Test-Path $CheckpointPath) -and -not $Resume) {
        Write-Host "Checkpoint for Subject $i already exists. Skipping training..."
        Write-Host ""
        continue
    }
    
    $TrainArgs = @(
        "-m", "src.train",
        "--modality", $Modality,
        "--subject", "$i"
    )

    if ($Epochs) {
        $TrainArgs += @("--epochs", $Epochs)
    }
    if ($Resume) {
        $TrainArgs += @("--resume")
    }
    if ($SharedOnly) {
        $TrainArgs += @("--shared-only")
    }
    if ($SharedManifest) {
        $TrainArgs += @("--shared-manifest", $SharedManifest)
    }
    
    & $PythonExe @TrainArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed for Subject $i. Exiting."
        exit $LASTEXITCODE
    }
    
    Write-Host "Finished training Subject $i."
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "All discovered ${ModalityUpper} subjects have been trained."
Write-Host "============================================================"

if (($Modality -eq "eeg") -and -not $SharedOnly) {
    Write-Host "Automatically generating the EEG summary table..."
    & $PythonExe "scripts\evaluate_eeg_table.py"
} else {
    Write-Host "Skipping automatic summary for ${ModalityUpper}."
    Write-Host "Use scripts/evaluate_retrieval.py for modality-vs-image metrics and scripts/evaluate_conversion.py for shared-image conversion."
}
