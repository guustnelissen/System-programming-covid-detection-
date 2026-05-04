#include "QT_basics.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QT_basics w;
	w.show();
	return a.exec();
}
