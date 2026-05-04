#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_ProjectXrayGUI.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class ProjectXrayGUI : public QMainWindow
{
	Q_OBJECT
public:
	ProjectXrayGUI(QWidget *parent = Q_NULLPTR);
	~ProjectXrayGUI();

private slots:
	void on_btnScan_clicked();
	void on_btnAnalyse_clicked();
	void on_sbLeeftijd_valueChanged(int value);

private:
	Ui::ProjectXrayGUIClass ui;

	Ort::Env env;
	Ort::Session* session;

	QString currentImagePath;
	QString lastLabel;

	QString getSurvivalChance(const std::string& label, int age);
};
