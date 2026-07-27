const { Presentation, PresentationFile } = await import('@oai/artifact-tool');

const W = 1024;
const H = 768;
const COLORS = {
  bg: '#F3F6FB',
  header: '#0B2545',
  panel: '#FFFFFF',
  border: '#C9D4E5',
  title: '#FFFFFF',
  body: '#111827',
  sub: '#334155',
  accent: '#0E7490',
  good: '#166534'
};

const FONT_TITLE = 'Calibri';
const FONT_BODY = 'Calibri';

function line(fill = COLORS.border, width = 1) {
  return { style: 'solid', fill, width };
}

function addBg(slide) {
  slide.shapes.add({
    geometry: 'rect',
    position: { left: 0, top: 0, width: W, height: H },
    fill: COLORS.bg,
    line: line(COLORS.bg, 0)
  });
}

function styleText(shape, { size = 20, bold = false, color = COLORS.body, align = 'left', valign = 'top' } = {}) {
  shape.text.fontSize = size;
  shape.text.bold = bold;
  shape.text.color = color;
  shape.text.typeface = FONT_BODY;
  shape.text.alignment = align;
  shape.text.verticalAlignment = valign;
  shape.text.insets = { left: 16, right: 16, top: 12, bottom: 12 };
}

function addHeader(slide, title, subtitle = '') {
  slide.shapes.add({
    geometry: 'rect',
    position: { left: 0, top: 0, width: W, height: 96 },
    fill: COLORS.header,
    line: line(COLORS.header, 0)
  });
  const t = slide.shapes.add({
    geometry: 'rect',
    position: { left: 28, top: 16, width: 800, height: 42 },
    fill: '#00000000',
    line: line('#00000000', 0)
  });
  t.text = title;
  t.text.typeface = FONT_TITLE;
  t.text.fontSize = 30;
  t.text.bold = true;
  t.text.color = COLORS.title;
  t.text.alignment = 'left';
  t.text.verticalAlignment = 'middle';

  if (subtitle) {
    const s = slide.shapes.add({
      geometry: 'rect',
      position: { left: 30, top: 58, width: 900, height: 28 },
      fill: '#00000000',
      line: line('#00000000', 0)
    });
    s.text = subtitle;
    s.text.typeface = FONT_BODY;
    s.text.fontSize = 15;
    s.text.bold = false;
    s.text.color = '#DDE7F7';
    s.text.alignment = 'left';
    s.text.verticalAlignment = 'middle';
  }
}

function addPanel(slide, x, y, w, h) {
  return slide.shapes.add({
    geometry: 'roundRect',
    position: { left: x, top: y, width: w, height: h },
    fill: COLORS.panel,
    line: line(COLORS.border, 1.2)
  });
}

function addText(slide, x, y, w, h, text, opts = {}) {
  const box = slide.shapes.add({
    geometry: 'rect',
    position: { left: x, top: y, width: w, height: h },
    fill: '#00000000',
    line: line('#00000000', 0)
  });
  box.text = text;
  styleText(box, opts);
  return box;
}

function addFooter(slide, n) {
  const f = slide.shapes.add({
    geometry: 'rect',
    position: { left: 0, top: H - 26, width: W, height: 26 },
    fill: '#E5ECF7',
    line: line('#E5ECF7', 0)
  });
  const tx = slide.shapes.add({
    geometry: 'rect',
    position: { left: W - 90, top: H - 24, width: 78, height: 20 },
    fill: '#00000000',
    line: line('#00000000', 0)
  });
  tx.text = `Slide ${n}`;
  tx.text.typeface = FONT_BODY;
  tx.text.fontSize = 12;
  tx.text.color = '#475569';
  tx.text.alignment = 'right';
  tx.text.verticalAlignment = 'middle';
}

function makeSlide(pres, n, title, subtitle, bodyText) {
  const s = pres.slides.add();
  addBg(s);
  addHeader(s, title, subtitle);
  addPanel(s, 36, 120, 952, 610);
  addText(s, 52, 140, 920, 574, bodyText, { size: 22, color: COLORS.body, valign: 'top' });
  addFooter(s, n);
  return s;
}

const p = Presentation.create({ slideSize: { width: W, height: H } });

// Slide 1
{
  const s = p.slides.add();
  addBg(s);
  addHeader(s, 'Confidence-Aware Hybrid Ensemble for Phishing Detection', 'Project presentation');
  addPanel(s, 64, 160, 896, 480);
  addText(s, 96, 205, 832, 120, 'Team Project\nIntelligent phishing website detection with robust hybrid decision logic', { size: 30, bold: true, align: 'center', valign: 'middle' });
  addText(s, 120, 350, 780, 170, '- Internal dataset: 11,055 samples\n- External dataset: 2,456 samples\n- Final system: Ensemble mode + Hybrid mode + Live API demo', { size: 22, color: COLORS.sub });
  addFooter(s, 1);
}

makeSlide(
  p,
  2,
  'Introduction and Objective',
  'Why this project matters',
  '- Phishing websites remain one of the most common cyber attack vectors.\n- Standard ML models can perform well internally but degrade on unseen data.\n- Objective: build a robust, deployable phishing detection system.\n- We target three goals:\n  1) strong in-distribution accuracy\n  2) better robustness under distribution shift\n  3) practical deployment via batch pipeline and API.'
);

makeSlide(
  p,
  3,
  'Literature and Gap',
  'Base paper plus recent direction',
  '- Base paper: EnLeM (2025) reports strong phishing detection with ensemble learning.\n- Recent studies use ANN, transformer, and hybrid methods.\n- Common limitation: external robustness and confidence-aware control are under-addressed.\n- Our gap focus: selective hybrid override only when model confidence is low.'
);

makeSlide(
  p,
  4,
  'Dataset and Features',
  'Data used in this project only',
  '- Training dataset: output.csv (from Training Dataset.arff), 11,055 rows, 30 features.\n- External dataset: .old.arff, 2,456 rows.\n- Feature values are discrete {-1, 0, 1}.\n- Feature harmonization on external data aligned 16 shared columns.\n- Example features: having_IP_Address, Prefix_Suffix, HTTPS_token, URL_of_Anchor, age_of_domain, Google_Index.'
);

makeSlide(
  p,
  5,
  'Proposed Method',
  'Confidence-aware hybrid ensemble',
  '- Model layer: Simple ANN + Deep ANN + Dropout-style ANN.\n- Aggregation: soft-voting ensemble probability.\n- Confidence score: C(x) = max(P(y=1|x), 1 - P(y=1|x)).\n- Hybrid logic: if confidence < 0.8 and security rules trigger, override prediction to phishing.\n- Two operation modes:\n  - Ensemble mode: balanced performance\n  - Hybrid mode: stronger phishing recall in uncertain cases.'
);

// Slide 6 novelty + contributions
makeSlide(
  p,
  6,
  'Novelty and Contributions',
  'What is new in our work',
  '- Confidence-aware selective override instead of always-on static rules.\n- Dual-mode architecture (ensemble and hybrid) for operational flexibility.\n- Explicit external validation under distribution shift, not only internal split.\n- End-to-end deployment artifacts: batch runner and live API for inference.'
);

// Slide 7 internal results
makeSlide(
  p,
  7,
  'Internal Results',
  'Strong in-distribution performance',
  '- 5-fold CV best model: ensemble_softvote\n  - F1 = 0.9704 +/- 0.0026\n  - ROC-AUC = 0.9954 +/- 0.0006\n- Internal run metrics (11,055 samples):\n  - Accuracy = 0.9809\n  - Precision = 0.9744\n  - Recall = 0.9917\n  - F1-score = 0.9830\n  - ROC-AUC = 0.9985'
);

// Slide 8 external base vs hybrid
{
  const s = p.slides.add();
  addBg(s);
  addHeader(s, 'External Results and Comparison', 'Base ensemble vs proposed hybrid');
  addPanel(s, 36, 120, 952, 610);

  addText(s, 52, 138, 920, 68, 'External dataset: 2,456 samples (.old.arff), phishing-positive scenario', { size: 20, bold: true, color: COLORS.sub });

  // table header row
  addPanel(s, 72, 220, 880, 56);
  addText(s, 86, 232, 180, 28, 'Method', { size: 18, bold: true });
  addText(s, 280, 232, 110, 28, 'Acc', { size: 18, bold: true, align: 'center' });
  addText(s, 400, 232, 110, 28, 'Prec', { size: 18, bold: true, align: 'center' });
  addText(s, 520, 232, 110, 28, 'Recall', { size: 18, bold: true, align: 'center' });
  addText(s, 640, 232, 110, 28, 'F1', { size: 18, bold: true, align: 'center' });
  addText(s, 760, 232, 150, 28, 'ROC-AUC', { size: 18, bold: true, align: 'center' });

  function row(y, method, acc, prec, rec, f1, auc, isHighlight = false) {
    addPanel(s, 72, y, 880, 52);
    addText(s, 86, y + 11, 180, 28, method, { size: 17, bold: isHighlight, color: isHighlight ? COLORS.good : COLORS.body });
    addText(s, 280, y + 11, 110, 28, acc, { size: 17, align: 'center' });
    addText(s, 400, y + 11, 110, 28, prec, { size: 17, align: 'center' });
    addText(s, 520, y + 11, 110, 28, rec, { size: 17, align: 'center' });
    addText(s, 640, y + 11, 110, 28, f1, { size: 17, align: 'center' });
    addText(s, 760, y + 11, 150, 28, auc, { size: 17, align: 'center' });
  }

  row(286, 'Base Ensemble', '0.8783', '0.9539', '0.8201', '0.8820', '0.9568');
  row(342, 'Proposed Hybrid', '0.8905', '0.9498', '0.8473', '0.8956', '0.9568', true);
  row(398, 'Delta (H - B)', '+0.0122', '-0.0041', '+0.0272', '+0.0137', '0.0000');

  addText(s, 72, 478, 880, 130, '- Key takeaway: hybrid logic improves recall and F1 on external data while keeping AUC stable.\n- This supports robustness under distribution shift.', { size: 20, color: COLORS.sub });

  addFooter(s, 8);
}

makeSlide(
  p,
  9,
  'System Demonstration Flow',
  'How to show live working in presentation',
  '- Step 1: Run batch pipeline (part15_end_to_end_system_runner.py).\n- Step 2: Show saved output CSV and summary JSON metrics.\n- Step 3: Start API server (part17_api_lite.py).\n- Step 4: Call /health then /predict using PowerShell request body from sample features.\n- Step 5: Show returned fields: mode, prediction_label, prediction_text, confidence, rules_triggered.'
);

makeSlide(
  p,
  10,
  'Industry Readiness',
  'Why this is deployable',
  '- CPU-friendly inference with lightweight API endpoint.\n- Supports security workflow integration (batch scanning + on-demand scoring).\n- Rule-trigger trace improves explainability for analysts.\n- External validation demonstrates practical reliability beyond internal-only testing.\n- Suitable for further hardening, monitoring, and retraining pipeline extension.'
);

makeSlide(
  p,
  11,
  'Conclusion and Next Steps',
  'Final message',
  '- We built a robust phishing detection pipeline using ANN ensemble + confidence-aware hybrid logic.\n- Internal performance is strong and external performance improves with hybrid mode.\n- Core novelty: selective override based on confidence and security rules.\n- Next steps: calibration, richer explainability, and continuous drift-aware updates.\n- Thank you.'
);

const outPath = 'C:/Users/PRIYAMVADA NAMBIAR/OneDrive - Amrita Vishwa Vidyapeetham/Documents/New project/learning_steps/outputs/Phishing_Detection_Presentation_clean_final.pptx';
const file = await PresentationFile.exportPptx(p);
await file.save(outPath);
console.log(outPath);
