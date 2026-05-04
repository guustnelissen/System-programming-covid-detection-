#include "QT_basics.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <vector>

QT_basics::QT_basics(QWidget *parent)
	: QMainWindow(parent),
	env(ORT_LOGGING_LEVEL_WARNING, "xray"),
	session(nullptr)
{
	ui.setupUi(this);

	// 1. Laad het AI model in (Zorg dat het pad klopt!)
	try {
		Ort::SessionOptions opts;
		session = new Ort::Session(env, L"C:/Users/lucie/OneDrive - Vrije Universiteit Brussel/BA 3 IR/SYSTEM PROGRAMMING/QT_basics/QT_basics/covid_model.onnx", opts);
	}
	catch (const std::exception& e) {
		QMessageBox::critical(this, "Fout", "Kan het AI-model niet laden!");
	}

	// 2. Koppel de leeftijd-verandering aan onze updatefunctie
	connect(ui.sbLeeftijd, QOverload<int>::of(&QSpinBox::valueChanged),
		this, &QT_basics::on_sbLeeftijd_valueChanged);
}

QT_basics::~QT_basics() {
	delete session; // Netjes het geheugen vrijmaken
}

// 1. AFBEELDING OPENEN
void QT_basics::on_btnScan_clicked()
{
	QString path = QFileDialog::getOpenFileName(
		this, "Kies een röntgenfoto", "C:\\", "Afbeeldingen (*.png *.jpg *.jpeg)"
	);

	if (path.isEmpty()) return;

	currentImagePath = path;
	QPixmap pix(path);

	// Toon de foto in je Designer label
	ui.lblFoto->setPixmap(pix.scaled(ui.lblFoto->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

	// Zet de knoppen en labels klaar
	ui.btnAnalyse->setEnabled(true);
	ui.lblResultaat->setText("Resultaat: -");
	ui.lblSurvival->setText("");
}

// 2. DE AI ANALYSE (CLASSIFY)
void QT_basics::on_btnAnalyse_clicked()
{
	if (currentImagePath.isEmpty()) return;

	// Laad de afbeelding in via OpenCV
	cv::Mat img = cv::imread(currentImagePath.toStdString());
	if (img.empty()) {
		QMessageBox::warning(this, "Fout", "Afbeelding kon niet worden geladen.");
		return;
	}

	// Preprocessing voor de AI (Schalen naar 299x299, RGB maken en normaliseren)
	cv::Mat resized;
	cv::resize(img, resized, cv::Size(299, 299));
	resized.convertTo(resized, CV_32F, 1.0 / 255.0);
	cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

	// Stop de data in een vector
	std::vector<float> inputData(1 * 299 * 299 * 3);
	std::memcpy(inputData.data(), resized.data, inputData.size() * sizeof(float));

	// Maak een Tensor voor ONNX
	auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<int64_t> inputShape = { 1, 299, 299, 3 };
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		memInfo, inputData.data(), inputData.size(),
		inputShape.data(), inputShape.size()
		);

	const char* inputNames[] = { "input_layer_1" };
	const char* outputNames[] = { "output_0" };

	// Voer het model uit
	auto outputs = session->Run(
		Ort::RunOptions{ nullptr },
		inputNames, &inputTensor, 1,
		outputNames, 1
	);

	float* scores = outputs[0].GetTensorMutableData<float>();
	std::vector<std::string> labels = { "Covid", "Normal", "Viral Pneumonia", "Non-Covid" };

	// Zoek de hoogste score
	int bestIdx = 0;
	for (int i = 1; i < 4; i++) {
		if (scores[i] > scores[bestIdx]) bestIdx = i;
	}

	lastLabel = QString::fromStdString(labels[bestIdx]);

	// Toon het resultaat in de GUI
	QString result = QString("Diagnose: %1 (%2%)")
		.arg(lastLabel)
		.arg(scores[bestIdx] * 100, 0, 'f', 1);

	ui.lblResultaat->setText(result);

	// Bereken en toon de overlevingskans
	QString survivalText = getSurvivalChance(lastLabel.toStdString(), ui.sbLeeftijd->value());
	ui.lblSurvival->setText(survivalText);
}

// 3. SURVIVAL RATE BEREKENEN
QString QT_basics::getSurvivalChance(const std::string& label, int age)
{
	double survivalChance;

	if (label == "Normal" || label == "Non-Covid") {
		return "Overlevingskans: 100% (geen infectie)";
	}

	// Leeftijd-tabel
	if (age <= 18)       survivalChance = 99.9;
	else if (age <= 44)  survivalChance = 99.8;
	else if (age <= 54)  survivalChance = 99.4;
	else if (age <= 64)  survivalChance = 98.0;
	else if (age <= 74)  survivalChance = 94.0;
	else if (age <= 84)  survivalChance = 87.0;
	else                 survivalChance = 75.0;

	if (label == "Viral Pneumonia") {
		survivalChance = qMin(99.9, survivalChance + 2.0);
	}

	return QString("Overlevingskans: %1%").arg(survivalChance, 0, 'f', 1);
}

// 4. ALS DE LEEFTIJD VERANDERT
void QT_basics::on_sbLeeftijd_valueChanged(int value)
{
	if (!lastLabel.isEmpty()) {
		QString survivalText = getSurvivalChance(lastLabel.toStdString(), value);
		ui.lblSurvival->setText(survivalText);
	}
}
