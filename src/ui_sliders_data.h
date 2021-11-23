#ifndef UI_SLIDERS_DATA_H
#define UI_SLIDERS_DATA_H

struct UiSlidersData
{
    float alpha;
    float sigma;

    UiSlidersData() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        alpha = 0.001f;
        sigma = 1.1f;
    }
};

#endif // #ifndef UI_SLIDERS_DATA_H
