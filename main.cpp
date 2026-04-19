#include "ProjectXrayGUI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);
	ProjectXrayGUI window;
	window.show();
	return app.exec();
}