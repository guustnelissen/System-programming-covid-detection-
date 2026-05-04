#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QT_basics.h"
#include <onnxruntime_cxx_api.h> // Voor het AI model
#include <opencv2/opencv.hpp>    // Voor het inladen van de foto

class QT_basics : public QMainWindow
{
	Q_OBJECT

public:
	QT_basics(QWidget *parent = Q_NULLPTR);
	~QT_basics(); // Destructor om het model netjes af te sluiten

private slots:
	void on_btnScan_clicked();
	void on_btnAnalyse_clicked();
	void on_sbLeeftijd_valueChanged(int value); // Als de leeftijd verandert

private:
	Ui::QT_basicsClass ui;

	// AI & ONNX Variabelen
	Ort::Env env;
	Ort::Session* session;

	// Opslag voor de foto en het resultaat
	QString currentImagePath;
	QString lastLabel;

	// Helper functie voor de overlevingskans
	QString getSurvivalChance(const std::string& label, int age);
};
