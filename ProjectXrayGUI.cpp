#include "ProjectXrayGUI.h"
#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include <vector>

ProjectXrayGUI::ProjectXrayGUI(QWidget* parent) //the constructor is called
	: QMainWindow(parent),
	env(ORT_LOGGING_LEVEL_WARNING, "xray"),
	session(nullptr)
{
	
	Ort::SessionOptions opts;
	session = new Ort::Session(env, L"C:\\covid_model.onnx", opts); //the trained model is loaded
	//new makes space free on the heap
	//The session pointer keeps the model in memory as long as the window is open.
	// UI opbouwen
	QWidget* central = new QWidget(this);
	setCentralWidget(central);

	QVBoxLayout* mainLayout = new QVBoxLayout(central); //VBox means vertically stacked

	// Afbeelding display
	imageLabel = new QLabel("No image loaded");
	imageLabel->setAlignment(Qt::AlignCenter);
	imageLabel->setMinimumSize(400, 400);
	imageLabel->setStyleSheet("border: 2px solid gray;");
	mainLayout->addWidget(imageLabel);

	// Resultaat label
	resultLabel = new QLabel("Result: -");
	resultLabel->setAlignment(Qt::AlignCenter);
	resultLabel->setStyleSheet("font-size: 18px; font-weight: bold;");
	mainLayout->addWidget(resultLabel);

	// Knoppen
	QHBoxLayout* btnLayout = new QHBoxLayout();
	openButton = new QPushButton("Open image");
	classifyButton = new QPushButton("Classify");
	classifyButton->setEnabled(false);
	btnLayout->addWidget(openButton);
	btnLayout->addWidget(classifyButton);
	mainLayout->addLayout(btnLayout);

	connect(openButton, &QPushButton::clicked, this, &ProjectXrayGUI::openImage); //this is where you connect the clicked buttons (signals) with a function 
	connect(classifyButton, &QPushButton::clicked, this, &ProjectXrayGUI::classify);

	setWindowTitle("X-Ray COVID Detector");
	resize(500, 550);
}

ProjectXrayGUI::~ProjectXrayGUI() {
	delete session;
}

void ProjectXrayGUI::openImage() {
	QString path = QFileDialog::getOpenFileName(  //opens the Windows file selection window
		this, "Choose a X-ray image", "C:\\",
		"Images (*.png *.jpg *.jpeg)"
	);
	if (path.isEmpty()) return;

	currentImagePath = path;
	QPixmap pix(path);		//loads image
	imageLabel->setPixmap(pix.scaled(400, 400, Qt::KeepAspectRatio));
	classifyButton->setEnabled(true);
	resultLabel->setText("Result: -");
}

void ProjectXrayGUI::classify() {
	cv::Mat img = cv::imread(currentImagePath.toStdString()); //loading image with opencv, a image is just a matrix of pixels
	if (img.empty()) {
		QMessageBox::warning(this, "Fault", "Image could not be loaded.");
		return;
	}

	// Preprocessing
	cv::Mat resized;
	cv::resize(img, resized, cv::Size(299, 299)); //resize the images to this format because the model is trained for that format
	resized.convertTo(resized, CV_32F, 1.0 / 255.0); //normalisation of the pixel values
	cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB); //the onnx model expects RGB but opencv loads it as BGR

	std::vector<float> inputData(1 * 299 * 299 * 3);
	std::memcpy(inputData.data(), resized.data, inputData.size() * sizeof(float));

	auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<int64_t> inputShape = { 1, 299, 299, 3 };
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		memInfo, inputData.data(), inputData.size(),
		inputShape.data(), inputShape.size()
		);

	const char* inputNames[] = { "input_layer_1" };
	const char* outputNames[] = { "output_0" };

	auto outputs = session->Run(
		Ort::RunOptions{ nullptr },
		inputNames, &inputTensor, 1,
		outputNames, 1
	);

	float* scores = outputs[0].GetTensorMutableData<float>();
	std::vector<std::string> labels = { "Covid", "Normal", "Viral Pneumonia", "Non-Covid" };

	int bestIdx = 0;
	for (int i = 1; i < 4; i++) {
		if (scores[i] > scores[bestIdx]) bestIdx = i;
	}

	QString result = QString("Diagnose: %1 (%2%)")
		.arg(QString::fromStdString(labels[bestIdx]))
		.arg(scores[bestIdx] * 100, 0, 'f', 1);

	resultLabel->setText(result);
}