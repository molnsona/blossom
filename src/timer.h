
#ifndef TIMER_H
#define TIMER_H

#include <chrono>

struct Timer {
	using timepoint = std::chrono::time_point<std::chrono::steady_clock>;

	float frametime;
	timepoint last_tick;

	Timer() {
		tick();
		frametime = 0.0;
	}

	void tick() {
		timepoint now = std::chrono::steady_clock::now();
		frametime = std::chrono::duration<float>(now-last_tick).count();
		last_tick=now;
	}
};

#endif
