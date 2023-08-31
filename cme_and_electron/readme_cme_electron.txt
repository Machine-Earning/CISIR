Electron + CME combined dataset

- features 
	- CME features
		1. donki_date: The date time from the DONKI catalog.
		2. cdaw_date: The date time from the CDAW catalog.
		3. 2nd_order_speed_20R: The 2nd order speed at 20 SR pulled from the CDAW catalog.
		4. latitude: The latitude pulled from the DONKI catalog.
		5. longitude: The longitude pulled from the DONKI catalog.
		6. donki_ha: The half angle pulled from the DONKI catalog.
		7. donki_speed: The linear speed pulled from the DONKI catalog.
		8. solar_wind_speed: The solar wind speed calculated from lulu.
		9. Type_2_Area: The Type II radio area calculated from CDAW.
		10. Accel: The acceleration pulled from the CDAW catalog.
		11. Central_PA: The Central position angle pulled from the CDAW catalog.
		12. V log V: The donki_speed * ln(donki_speed) using the respective donki_speed values.
		13. MPA: The MPA pulled from the CDAW catalog.
		14. 2nd_order_speed_final: The 2nd order speed final pulled from the CDAW catalog.
		15. 2nd_order_speed_initial: The 2nd order speed initial pulled from the CDAW catalog.
		16. connection_angle_no_sign: The calculated connection angle without a sign.
		17. connection_angle: The calculated connection angle in radians.
		18. connection_angle_degrees: The calculated connection angle in degrees.
		19. connection_angle_no_sign_opposite: The inverse of the connection angle without a sign.
		20. richardson_formula_1.0_c ... richardson_formula_2.0_c: Experimentation with varying a value inside of the Richardson formula.
		21. log_richardson_formula_1.0_c ... log_richardson_formula_2.0_c: The log of the Richardson formula.
		22. richardson_formula_hw_0_degrees ... richardson_formula_10_degrees: More experimentation with the Richardson formula.
		23. log_richardson_formula_hw_0_degrees ... log_richardson_formula_10_degrees: The log of the Richardson formula.
		24. richardson_formula_degrees: The Richardson formula calculation in degrees.
		26. CMEs_past_month: A history feature calculated based on the number of CMEs in this dataset in the past month.
		27. CMEs_past_9_hours: A history feature calculated based on the number of CMEs in this dataset in the past 9 hours.
		29. Max_speed_past_day: A history feature calculated based on the maximum donki_speed of CMEs occurring in this dataset in the past day.
		30. CMEs_over_1000_past_9_hrs: A history feature calculated based on the number of CMEs in this dataset in the past 9 hours with a donki_speed >= 1000.
		31. sunspots: The daily sunspot count.

	- Electron Intensity (past 2 hours)
		- electron (past 2 hours): natural log of electron flux from >0.25 MeV channel
		- electron_high (past 2 hours): natural log of electron flux from >0.67 MeV channel
		? Not other features from here ?

- target
	- 10 MeV proton intensity 
	? where are they coming from exactly ?