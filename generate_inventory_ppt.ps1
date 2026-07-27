$ErrorActionPreference = "Stop"

function RGBInt([int]$r, [int]$g, [int]$b) {
    return [int]($r + ($g -shl 8) + ($b -shl 16))
}

$msoFalse = 0
$msoTrue = -1
$ppLayoutBlank = 12
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoShapeOval = 9
$msoShapeRightArrow = 33
$msoSendToBack = 1
$ppSlideSizeOnScreen16x9 = 16

$cBg = RGBInt 11 15 23
$cPanel = RGBInt 18 26 39
$cPanelSoft = RGBInt 23 34 49
$cWhite = RGBInt 244 248 255
$cMuted = RGBInt 143 163 191
$cLine = RGBInt 38 50 68
$cAccent = RGBInt 34 211 238
$cAccent2 = RGBInt 103 232 167
$cWarn = RGBInt 250 204 21

function Add-Background($slide, $w, $h) {
    $bg = $slide.Shapes.AddShape($msoShapeRectangle, 0, 0, $w, $h)
    $bg.Fill.ForeColor.RGB = $cBg
    $bg.Line.Visible = $msoFalse
    $bg.ZOrder($msoSendToBack) | Out-Null

    $orb1 = $slide.Shapes.AddShape($msoShapeOval, -120, -120, 320, 320)
    $orb1.Fill.ForeColor.RGB = $cAccent
    $orb1.Fill.Transparency = 0.90
    $orb1.Line.Visible = $msoFalse
    $orb1.ZOrder($msoSendToBack) | Out-Null

    $orb2 = $slide.Shapes.AddShape($msoShapeOval, $w - 220, $h - 220, 280, 280)
    $orb2.Fill.ForeColor.RGB = $cAccent2
    $orb2.Fill.Transparency = 0.91
    $orb2.Line.Visible = $msoFalse
    $orb2.ZOrder($msoSendToBack) | Out-Null

    $bar = $slide.Shapes.AddShape($msoShapeRectangle, 0, 0, $w, 5)
    $bar.Fill.ForeColor.RGB = $cAccent
    $bar.Line.Visible = $msoFalse
}

function Add-TextBox($slide, [string]$text, [single]$x, [single]$y, [single]$w, [single]$h, [int]$size, [bool]$bold, [int]$color, [int]$align = 1) {
    $box = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $box.TextFrame.TextRange.Text = $text
    $box.TextFrame.MarginLeft = 2
    $box.TextFrame.MarginRight = 2
    $box.TextFrame.MarginTop = 2
    $box.TextFrame.MarginBottom = 2
    $box.TextFrame.WordWrap = $msoTrue
    $box.TextFrame.TextRange.Font.Name = "Segoe UI"
    $box.TextFrame.TextRange.Font.Size = $size
    $box.TextFrame.TextRange.Font.Bold = $(if ($bold) { $msoTrue } else { $msoFalse })
    $box.TextFrame.TextRange.Font.Color.RGB = $color
    $box.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $box
}

function Add-Card($slide, [single]$x, [single]$y, [single]$w, [single]$h, [string]$title, [string]$body) {
    $card = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, $w, $h)
    $card.Fill.ForeColor.RGB = $cPanel
    $card.Line.ForeColor.RGB = $cLine
    $card.Line.Weight = 1
    $card.Adjustments.Item(1) = 0.08
    [void](Add-TextBox $slide $title ($x + 14) ($y + 10) ($w - 24) 26 16 $true $cWhite)
    [void](Add-TextBox $slide $body ($x + 14) ($y + 42) ($w - 24) ($h - 50) 12 $false $cMuted)
}

function Add-Chip($slide, [string]$text, [single]$x, [single]$y, [single]$w = 390, [single]$h = 34, [int]$fillColor = 0) {
    if ($fillColor -eq 0) { $fillColor = $cPanelSoft }
    $chip = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, $w, $h)
    $chip.Fill.ForeColor.RGB = $fillColor
    $chip.Line.ForeColor.RGB = $cLine
    $chip.Adjustments.Item(1) = 0.2
    [void](Add-TextBox $slide $text ($x + 12) ($y + 7) ($w - 20) 22 13 $true $cWhite)
}

function Add-TitleBlock($slide, [string]$title, [string]$subtitle = "") {
    [void](Add-TextBox $slide $title 44 32 870 76 36 $true $cWhite)
    if ($subtitle -ne "") {
        [void](Add-TextBox $slide $subtitle 44 90 860 34 16 $false $cMuted)
    }
}

function Add-FlowStep($slide, [single]$x, [single]$y, [single]$w, [single]$h, [string]$label, [int]$isAccent = 0) {
    $step = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, $w, $h)
    $step.Fill.ForeColor.RGB = $(if ($isAccent -eq 1) { $cPanelSoft } else { $cPanel })
    $step.Line.ForeColor.RGB = $cLine
    $step.Adjustments.Item(1) = 0.12
    [void](Add-TextBox $slide $label ($x + 8) ($y + 20) ($w - 16) 28 12 $true $cWhite 2)
}

function Add-FlowArrow($slide, [single]$x, [single]$y, [single]$w, [single]$h) {
    $arrow = $slide.Shapes.AddShape($msoShapeRightArrow, $x, $y, $w, $h)
    $arrow.Fill.ForeColor.RGB = $cAccent
    $arrow.Line.Visible = $msoFalse
    $arrow.Fill.Transparency = 0.10
}

$outputPath = Join-Path $PWD "AI_Smart_Inventory_Management_System.pptx"
$dashboardPath = Join-Path $PWD "dashboard_mock.png"

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = $msoTrue
$presentation = $ppt.Presentations.Add()
try {
    $presentation.PageSetup.SlideSize = $ppSlideSizeOnScreen16x9
} catch {
    # Fallback to default slide size if enum is unavailable.
}
$slideW = 960
$slideH = 540

# Slide 1
$s = $presentation.Slides.Add(1, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "AI Smart Inventory Management System" "Demand Forecasting using Machine Learning"
[void](Add-Chip $s "Team Members: Name 1 | Name 2 | Name 3 | Name 4" 44 164 520 38)
Add-Card $s 44 228 420 220 "Project Scope" "End-to-end ML workflow for inventory demand forecasting, deployment, and MLOps."
Add-Card $s 492 228 424 220 "Technology Stack" "Python, Random Forest, XGBoost, FastAPI, Streamlit, Docker, GitHub Actions"

# Slide 2
$s = $presentation.Slides.Add(2, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Problem Statement"
Add-Card $s 44 146 872 244 "Key Challenges" "Overstocking -> high cost`nUnderstocking -> lost sales`nNo accurate demand prediction"
[void](Add-Chip $s "Retail needs intelligent demand forecasting" 44 418 520 38 $cAccent)

# Slide 3
$s = $presentation.Slides.Add(3, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Solution"
Add-Card $s 44 156 274 250 "ML-based Prediction" "Forecast demand with supervised learning to support better stock decisions."
Add-Card $s 343 156 274 250 "Feature-rich Inputs" "Uses historical sales and engineered features to improve predictive power."
Add-Card $s 642 156 274 250 "Automated E2E Pipeline" "Integrated workflow from training to deployment and monitoring."

# Slide 4
$s = $presentation.Slides.Add(4, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Dataset" "Time-series retail data"
Add-Card $s 44 160 200 200 "Store" "Store-level sales context"
Add-Card $s 264 160 200 200 "Product Family" "Category-level behavior"
Add-Card $s 484 160 200 200 "Sales" "Historical target signal"
Add-Card $s 704 160 212 200 "Promotion" "Demand uplift driver"
[void](Add-Chip $s "Dataset versioned using DVC" 44 400 330 38 $cPanelSoft)

# Slide 5
$s = $presentation.Slides.Add(5, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Preprocessing"
Add-Card $s 44 160 280 220 "Data Quality" "Missing values handled"
Add-Card $s 340 160 280 220 "Time Consistency" "Sorted by time"
Add-Card $s 636 160 280 220 "Encoding" "Categorical encoding done"
[void](Add-Chip $s "Same preprocessing used in training & inference" 44 410 520 38 $cAccent2)

# Slide 6
$s = $presentation.Slides.Add(6, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Feature Engineering"
Add-Card $s 44 150 210 216 "Lag Features" "Captures past sales behavior"
Add-Card $s 272 150 210 216 "Rolling Mean" "Represents local trend"
Add-Card $s 500 150 210 216 "Rolling Std" "Measures demand volatility"
Add-Card $s 728 150 188 216 "Time Features" "Seasonality signal"
[void](Add-Chip $s "Core strength of the model" 44 396 310 38 $cAccent)

# Slide 7
$s = $presentation.Slides.Add(7, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Models Used"
Add-Card $s 120 176 320 220 "Random Forest" "Robust baseline with ensemble trees"
Add-Card $s 520 176 320 220 "XGBoost" "Boosted trees for stronger non-linear learning"
[void](Add-Chip $s "Compared using RMSE" 360 420 240 38 $cPanelSoft)

# Slide 8
$s = $presentation.Slides.Add(8, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Results"
[void](Add-Chip $s "XGBoost -> best performance" 44 126 360 34 $cAccent2)
[void](Add-Chip $s "Lower RMSE" 420 126 180 34 $cPanelSoft)
[void](Add-Chip $s "Handles non-linear patterns better" 616 126 300 34 $cPanelSoft)

$chartBox = $s.Shapes.AddShape($msoShapeRoundedRectangle, 120, 190, 720, 250)
$chartBox.Fill.ForeColor.RGB = $cPanel
$chartBox.Line.ForeColor.RGB = $cLine
$rfBar = $s.Shapes.AddShape($msoShapeRoundedRectangle, 190, 275, 460, 38)
$rfBar.Fill.ForeColor.RGB = RGBInt 248 113 113
$rfBar.Line.Visible = $msoFalse
[void](Add-TextBox $s "Random Forest RMSE (higher)" 194 248 250 20 11 $false $cMuted)
$xgBar = $s.Shapes.AddShape($msoShapeRoundedRectangle, 190, 340, 350, 38)
$xgBar.Fill.ForeColor.RGB = $cAccent2
$xgBar.Line.Visible = $msoFalse
[void](Add-TextBox $s "XGBoost RMSE (lower)" 194 313 220 20 11 $false $cMuted)

# Slide 9
$s = $presentation.Slides.Add(9, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Pipeline Architecture"

$startX = 28
$stepY = 230
$stepW = 150
$stepH = 80
$gap = 36
$arrowW = 26

Add-FlowStep $s $startX $stepY $stepW $stepH "Data"
Add-FlowArrow $s ($startX + $stepW + 5) ($stepY + 26) $arrowW 28
Add-FlowStep $s ($startX + ($stepW + $gap) * 1) $stepY $stepW $stepH "Preprocess"
Add-FlowArrow $s ($startX + ($stepW + $gap) * 1 + $stepW + 5) ($stepY + 26) $arrowW 28
Add-FlowStep $s ($startX + ($stepW + $gap) * 2) $stepY $stepW $stepH "Features"
Add-FlowArrow $s ($startX + ($stepW + $gap) * 2 + $stepW + 5) ($stepY + 26) $arrowW 28
Add-FlowStep $s ($startX + ($stepW + $gap) * 3) $stepY $stepW $stepH "Train"
Add-FlowArrow $s ($startX + ($stepW + $gap) * 3 + $stepW + 5) ($stepY + 26) $arrowW 28
Add-FlowStep $s ($startX + ($stepW + $gap) * 4) $stepY $stepW $stepH "Save Model" 1
[void](Add-Chip $s "Fully automated using pipeline.py" 44 408 360 38 $cAccent)

# Slide 10
$s = $presentation.Slides.Add(10, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Inference Pipeline"

$startX = 80
$stepY = 220
$stepW = 170
$stepH = 90
$gap = 48
$arrowW = 32

Add-FlowStep $s $startX $stepY $stepW $stepH "Input"
Add-FlowArrow $s ($startX + $stepW + 8) ($stepY + 30) $arrowW 30
Add-FlowStep $s ($startX + ($stepW + $gap) * 1) $stepY $stepW $stepH "Preprocess"
Add-FlowArrow $s ($startX + ($stepW + $gap) * 1 + $stepW + 8) ($stepY + 30) $arrowW 30
Add-FlowStep $s ($startX + ($stepW + $gap) * 2) $stepY $stepW $stepH "Features"
Add-FlowArrow $s ($startX + ($stepW + $gap) * 2 + $stepW + 8) ($stepY + 30) $arrowW 30
Add-FlowStep $s ($startX + ($stepW + $gap) * 3) $stepY $stepW $stepH "Predict" 1
[void](Add-Chip $s "Same logic reused -> no data leakage" 44 408 410 38 $cAccent2)

# Slide 11
$s = $presentation.Slides.Add(11, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "FastAPI Deployment"
Add-Card $s 44 164 430 242 "API Contract" "Endpoint: /predict`nJSON input`nReturns prediction`nSchema validation using Pydantic"

$jsonCard = $s.Shapes.AddShape($msoShapeRoundedRectangle, 500, 164, 416, 242)
$jsonCard.Fill.ForeColor.RGB = RGBInt 13 22 35
$jsonCard.Line.ForeColor.RGB = $cLine
$jsonCard.Adjustments.Item(1) = 0.07
[void](Add-TextBox $s "{`n  ""store"": 12,`n  ""family"": ""Dairy"",`n  ""promotion"": 1`n}`n`n-> { ""prediction"": 1248.6 }" 518 184 380 210 13 $false $cAccent)

# Slide 12
$s = $presentation.Slides.Add(12, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "MLOps"
Add-Card $s 44 164 280 236 "Docker" "Containerization for reproducible deployment"
Add-Card $s 340 164 280 236 "Logging" "Tracking predictions and operational behavior"
Add-Card $s 636 164 280 236 "CI/CD" "Automation using GitHub Actions"
[void](Add-Chip $s "This slide gives extra marks" 44 420 290 38 $cWarn)

# Slide 13
$s = $presentation.Slides.Add(13, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Dashboard"
[void](Add-Chip $s "Streamlit SaaS UI | KPI cards | Graph + prediction" 44 118 530 34 $cPanelSoft)

$frame = $s.Shapes.AddShape($msoShapeRoundedRectangle, 44, 160, 872, 334)
$frame.Fill.ForeColor.RGB = $cPanel
$frame.Line.ForeColor.RGB = $cLine
$frame.Adjustments.Item(1) = 0.03

if (Test-Path $dashboardPath) {
    [void]$s.Shapes.AddPicture($dashboardPath, $msoFalse, $msoTrue, 58, 174, 844, 306)
} else {
    [void](Add-TextBox $s "Screenshot here" 390 305 200 30 18 $true $cWhite 2)
}

# Slide 14
$s = $presentation.Slides.Add(14, $ppLayoutBlank)
Add-Background $s $slideW $slideH
[void](Add-TextBox $s "Live Demo" 0 210 960 80 54 $true $cWhite 2)
[void](Add-TextBox $s "Switching to the running application" 0 286 960 30 16 $false $cMuted 2)

# Slide 15
$s = $presentation.Slides.Add(15, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Conclusion"
Add-Card $s 84 180 250 220 "End-to-end ML system" "From raw retail data to deployed predictions"
Add-Card $s 354 180 250 220 "Production-ready" "FastAPI service + Streamlit interface + MLOps"
Add-Card $s 624 180 250 220 "Scalable" "Pipeline-driven design supports growth and reuse"

# Slide 16
$s = $presentation.Slides.Add(16, $ppLayoutBlank)
Add-Background $s $slideW $slideH
Add-TitleBlock $s "Future Work"
Add-Card $s 80 180 250 220 "External Signals" "Add weather and holiday data"
Add-Card $s 354 180 250 220 "Advanced Models" "Use deep learning (LSTM)"
Add-Card $s 628 180 250 220 "Real-time Integration" "Connect with live inventory systems"

$presentation.SaveAs($outputPath)
$presentation.Close()
$ppt.Quit()

[System.Runtime.Interopservices.Marshal]::ReleaseComObject($presentation) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
[GC]::Collect()
[GC]::WaitForPendingFinalizers()

Write-Output "Created: $outputPath"
