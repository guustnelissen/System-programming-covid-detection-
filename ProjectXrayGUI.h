#pragma once
#include <QtWidgets/QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QString>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class ProjectXrayGUI : public QMainWindow {  //this is inheritance of a class
	Q_OBJECT		// a Qt macro that makes signals and slots possible
					// slots =  functions thats are linked with buttons
public:
	ProjectXrayGUI(QWidget* parent = nullptr);
	~ProjectXrayGUI();

private slots:
	void openImage();
	void classify();

private:
	QLabel* imageLabel;
	QLabel* resultLabel;
	QPushButton* openButton;
	QPushButton* classifyButton;
	QString currentImagePath;

	Ort::Env env;
	Ort::Session* session;
};
