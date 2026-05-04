#include "ProjectXrayGUI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ProjectXrayGUI w;
	w.show();
	return a.exec();
}
