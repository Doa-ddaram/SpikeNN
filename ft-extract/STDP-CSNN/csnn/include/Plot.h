#ifndef _PLOT_H
#define _PLOT_H

#ifdef ENABLE_QT

#include <QWidget>

class Plot : public QWidget {

public:
	Plot() : QWidget() {

	}

	virtual void initialize() {

	}

	virtual void on_tick() = 0;
	virtual void on_refresh() = 0;
};

#endif

#endif
