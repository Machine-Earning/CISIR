# Headings explained: DONKI_CDAW_all_features_unnormalized_removed_CMEs.csv

## donki_date
The date time from the DONKI catalog

## cdaw_date
The date time from the CDAW catalog

## ESP
ESP = Energetic storm particle. I'm not sure what this feature means or why they are all 0. See Thomas's and Tarsoly's papers for more details

## 2nd_order_speed_20R
The 2nd order speed at 20 SR pulled from the CDAW catalog

## latitude
The latitude pulled from the DONKI catalog

## longitude
The longitude pulled from the DONKI catalog

## donki_ha
The half angle pulled from the DONKI catalog

## donki_speed
The linear speed pulled from the DONKI catalog

## solar_wind_speed
The solar wind speed calculated from lulu. Can be found: https://omniweb.gsfc.nasa.gov/form/dx1.html.

## Type_2_Area
The Type II radio area calculated from CDAW: https://cdaw.gsfc.nasa.gov/CME_list/radio/waves_type2.html

## Accel
The acceleration pulled from the CDAW catalog

## Central_PA
The Central position angle pulled from the CDAW catalog

## V log V
The donki_speed * ln(donki_speed) using the respective donki_speed values.

## MPA
The MPA pulled from the CDAW catalog

## 2nd_order_speed_final
The 2nd order speed final pulled from the CDAW catalog

## 2nd_order_speed_initial
The 2nd order speed initial pulled from the CDAW catalog

## connection_angle_no_sign
The calculated connection angle without a sign (basically the absolute value)

## connection_angle
The calculated connection angle in radians

## connection_angle_degrees
The calculated connection angle in degrees

## connection_angle_no_sign_opposite
I don't completely remember but I believe it's exactly what it sounds like, the inverse of the connection angle without a sign. It appears that connection_angle_no_sign_opposite + connection_angle_no_sign = PI

## richardson_formula_1.0_c ... richardson_formula_2.0_c
I also don't completely remember but they were experimenting with varying a value inside of the Richardson formula. The Richardson formula is I = 0.013 * exp(0.00036 * V - phi^2 / (2 * sigma^2)), sigma = 43, V is speed, phi is connection angle.
At first I thought that maybe the "c" part referred to the constant 0.013, but upon further analysis I think it makes more sense that they were messing around with changing the coefficient in front of the connection_angle (the phi term). In the formula, it's a fixed -1. I think they tried values from 1.0 to 2.0.

## log_richardson_formula_1.0_c ... log_richardson_formula_2.0_c
You can take the log of both sides of the richardson formula and then they varied the same c value from 1.0 to 2.0. See above about what the c from 1.0 to 2.0 means.

## richardson_formula_hw_0_degrees richardson_formula_w_0_degrees ... richardson_formula_hw_10_degrees richardson_formula_w_10_degrees richardson_formula_10_degrees
More experimentation with the Richardson formula. Here I think they were messing around with the calculation of the connection angle. The connection angle depends on the DONKI latitude, DONKI longitude, solar wind speed (or fix at the typical 57), and a value that varies between -7 and 7 depending on the season which we have all been setting to 0 for ease of implementation. I believe they were experimenting with changing that fixed at 0 value to either 0, 5, or 10. The hw sounds like half width and the regular w sounds like width.

Alternatively, they could be messing with the Gaussian width (the sigma in the formula) and varying it by 0 i.e. original 43, 5 (48), and 10 (53). I'm not 100% sure.

## log_richardson_formula_hw_0_degrees log_richardson_formula_w_0_degrees ... log_richardson_formula_hw_10_degrees log_richardson_formula_w_10_degrees log_richardson_formula_10_degrees
Same as above except for LN of the richardson formula.

## richardson_formula_degrees
The richardson formula calculation in degrees

## V^V^2_replacement
I believe the V(V^2) mentioned in Appendix A of Torres's paper.

## CMEs_past_month
A history feature calculated based on the number of CMEs in this dataset in the past month

## CMEs_past_9_hours
A history feature calculated based on the number of CMEs in this dataset in the past 9 hours

## DONKI_double_CME
A hand-picked annotated feature by Dr. Zhang in which he analyzes specific SEP events that may be caused by more than 1 CME calling them Double CME. In my dataset (not this one), I know for certain he did not analyze all of the events. In this dataset, I believe this feature analysis is also incomplete for all events.

## Max_speed_past_day
A history feature calculated based on the maximum donki_speed of CMEs occuring in this dataset in the past day

## CMEs_over_1000_past_9_hrs
A history feature calculated based on the number of CMEs in this dataset in the past 9 hours with a donki_speed >= 1000

## sunspots
The daily sunspot count pulled from: https://www.sidc.be/SILSO/infosndtot

## flare_intensity
This should be the peak intensity value

## is_ESP
Related to the ESP. See Thomas's and Tarsoly's papers for more details

## ESP_formula
Related to the ESP. See Thomas's and Tarsoly's papers for more details

## only_longitude_ESP
Related to the ESP. See Thomas's and Tarsoly's papers for more details

## is_ESP_hw
Related to the ESP. See Thomas's and Tarsoly's papers for more details

## speed_times_HW
I believe it's what it sounds like: donki_speed * donki_hw

## target
The target classification: 1 = SEP, 0 = Background (typically also 2 = Elevated)
