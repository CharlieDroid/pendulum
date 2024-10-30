/****************************************************************************************************************************
  Portenta_H7_ISR_Timer.h
  For Portenta_H7 boards
  Written by Khoi Hoang

  Built by Khoi Hoang https://github.com/khoih-prog/Portenta_H7_TimerInterrupt
  Licensed under MIT license

  Now even you use all these new 16 ISR-based timers,with their maximum interval practically unlimited (limited only by
  unsigned long miliseconds), you just consume only one Portenta_H7 STM32 timer and a1void conflicting with other cores' tasks.
  The accuracy is nearly perfect compared to software timers. The most important feature is they're ISR-based timers
  Therefore, their executions are not blocked by bad-behaving functions / tasks.
  This important feature is absolutely necessary for mission-critical tasks.

  Version: 1.4.0

  Version Modified By   Date      Comments
  ------- -----------  ---------- -----------
  1.2.1   K.Hoang      15/09/2021 Initial coding for Portenta_H7
  1.3.0   K.Hoang      17/09/2021 Add PWM features and examples
  1.3.1   K.Hoang      21/09/2021 Fix warnings in PWM examples
  1.4.0   K.Hoang      22/01/2022 Fix `multiple-definitions` linker error. Fix bug
*****************************************************************************************************************************/

#pragma once

#ifndef ISR_TIMER_GENERIC_H
#define ISR_TIMER_GENERIC_H

#include "Portenta_H7_ISR_Timer.hpp"
#include "Portenta_H7_ISR_Timer-Impl.h"

#endif    // ISR_TIMER_GENERIC_H

