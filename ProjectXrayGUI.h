#pragma once
#include <QtWidgets/QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QString>
#include <QSpinBox>
#include <QLabel>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>


class ProjectXrayGUI : public QMainWindow {  //this is inheritance of a class
	Q_OBJECT		// a Qt macro that makes signals and slots possible
					// slots =  functions thats are linked with buttons
public:
	ProjectXrayGUI(QWidget* parent = nullptr);
	~ProjectXrayGUI();

private slots:
	void openImage();
	void classify();
	void updateSurvivalChance(); //so that the survival chance updates wheb you change the age

private:
	QLabel* imageLabel;
	QLabel* resultLabel;
	QPushButton* openButton;
	QPushButton* classifyButton;
	QString currentImagePath;
	QSpinBox* ageInput;
	QLabel* ageLabel;
	QLabel* survivalChanceLabel;
	QString getSurvivalChance(const std::string& label, int age);
	QString lastLabel;

	Ort::Env env;
	Ort::Session* session;
};
