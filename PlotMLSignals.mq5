//+------------------------------------------------------------------+
//| Indicator to plot ML signals from CSV for WIN futures           |
//| Place win_ml_signals.csv in MQL5\Files folder                   |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

input string SignalFile      = "win_ml_signals.csv";
input int    LookbackDays    = 1;     // how many days back to plot signals
input int    RefreshSeconds  = 15;    // how often to re-read the CSV
input bool   OnlyTransitions = true;  // true = only plot 0->1 buys / 1->0 sells (recommended for M5)

//---
int OnInit()
  {
   EventSetTimer(RefreshSeconds);
   PlotSignals(); // first attempt immediately
   return INIT_SUCCEEDED;
  }

void OnTimer()
  {
   PlotSignals();
  }

int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[], const double &close[],
                const long &tick_volume[], const long &volume[], const int &spread[])
  {
   if(prev_calculated == 0)
      PlotSignals();
   return rates_total;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, "BuySignal_");
   ObjectsDeleteAll(0, "SellSignal_");
   ObjectsDeleteAll(0, "SL_");
   ObjectsDeleteAll(0, "SLLine_");
  }

//---
void PlotSignals()
  {
   string sym = _Symbol;            // use the chart's symbol/timeframe
   ENUM_TIMEFRAMES tf = _Period;

   int barsAvail = Bars(sym, tf);
   datetime lastBarTime = iTime(sym, tf, 0);
   if(lastBarTime <= 0) lastBarTime = TimeCurrent();
   datetime cutoff = lastBarTime - LookbackDays * 86400;

   if(barsAvail < 10)
     {
      Print("Series not ready — will retry on next tick");
      return;
     }

   ObjectsDeleteAll(0, "BuySignal_");
   ObjectsDeleteAll(0, "SellSignal_");
   ObjectsDeleteAll(0, "SL_");
   ObjectsDeleteAll(0, "SLLine_");

   int fileHandle = FileOpen(SignalFile, FILE_READ|FILE_TXT|FILE_ANSI);
   if(fileHandle == INVALID_HANDLE)
     {
      Print("Failed to open ", SignalFile, " error=", GetLastError());
      return;
     }

   // Skip header line
   FileReadString(fileHandle);

   int buyCount = 0, sellCount = 0, totalRows = 0, inWindow = 0;
   int prevSignal = -1;

   while(!FileIsEnding(fileHandle))
     {
      string line = FileReadString(fileHandle);
      if(line == "") continue;
      string parts[];
      int    nParts = StringSplit(line, ',', parts);
      if(nParts < 2) continue;
      string timestamp = parts[0];
      string sigStr    = parts[1];
      string slStr     = nParts >= 3 ? parts[2] : "0";
      if(timestamp == "" || sigStr == "") continue;
      totalRows++;

      int      signal = (int)StringToInteger(sigStr);
      datetime dt     = (datetime)StringToInteger(timestamp);
      if(dt <= 0) continue;

      bool isBuy, isSell;
      if(OnlyTransitions)
        {
         isBuy  = (signal == 1 && prevSignal == 0);
         isSell = (signal == 0 && prevSignal == 1);
        }
      else
        {
         isBuy  = (signal == 1);
         isSell = (signal == 0);
        }
      prevSignal = signal;

      if(dt < cutoff) continue;
      inWindow++;

      if(!isBuy && !isSell) continue;

      int bar = iBarShift(sym, tf, dt, false);
      if(bar < 0) continue;
      datetime barTime = iTime(sym, tf, bar);

      double slPoints = StringToDouble(slStr);

      int    tfSec  = PeriodSeconds(tf);
      int    slBars  = 12;   // how many bars the SL line extends to the right

      if(isBuy)
        {
         string name       = "BuySignal_" + IntegerToString((int)dt);
         double arrowPrice = iLow(sym, tf, bar) * 0.9995;
         if(ObjectCreate(0, name, OBJ_ARROW, 0, barTime, arrowPrice))
           {
            ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 233);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrLime);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, name, OBJPROP_BACK, false);
            buyCount++;
            if(slPoints > 0)
              {
               // text label
               string slName = "SL_" + IntegerToString((int)dt);
               ObjectCreate(0, slName, OBJ_TEXT, 0, barTime, arrowPrice * 0.9991);
               ObjectSetString(0, slName, OBJPROP_TEXT, "SL:" + DoubleToString(slPoints, 0) + "p");
               ObjectSetInteger(0, slName, OBJPROP_COLOR, clrLime);
               ObjectSetInteger(0, slName, OBJPROP_FONTSIZE, 7);
               ObjectSetInteger(0, slName, OBJPROP_ANCHOR, ANCHOR_TOP);
               // horizontal SL line below entry close
               double slPrice  = iClose(sym, tf, bar) - slPoints;
               datetime time2  = barTime + tfSec * slBars;
               string lineName = "SLLine_" + IntegerToString((int)dt);
               if(ObjectCreate(0, lineName, OBJ_TREND, 0, barTime, slPrice, time2, slPrice))
                 {
                  ObjectSetInteger(0, lineName, OBJPROP_COLOR, clrLime);
                  ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_DASH);
                  ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
                  ObjectSetInteger(0, lineName, OBJPROP_RAY_LEFT,  false);
                  ObjectSetInteger(0, lineName, OBJPROP_RAY_RIGHT, false);
                  ObjectSetInteger(0, lineName, OBJPROP_BACK, true);
                 }
              }
           }
        }
      else
        {
         string name       = "SellSignal_" + IntegerToString((int)dt);
         double arrowPrice = iHigh(sym, tf, bar) * 1.0005;
         if(ObjectCreate(0, name, OBJ_ARROW, 0, barTime, arrowPrice))
           {
            ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 234);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrRed);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, name, OBJPROP_BACK, false);
            sellCount++;
            if(slPoints > 0)
              {
               // text label
               string slName = "SL_" + IntegerToString((int)dt);
               ObjectCreate(0, slName, OBJ_TEXT, 0, barTime, arrowPrice * 1.0009);
               ObjectSetString(0, slName, OBJPROP_TEXT, "SL:" + DoubleToString(slPoints, 0) + "p");
               ObjectSetInteger(0, slName, OBJPROP_COLOR, clrRed);
               ObjectSetInteger(0, slName, OBJPROP_FONTSIZE, 7);
               ObjectSetInteger(0, slName, OBJPROP_ANCHOR, ANCHOR_BOTTOM);
               // horizontal SL line above entry close
               double slPrice  = iClose(sym, tf, bar) + slPoints;
               datetime time2  = barTime + tfSec * slBars;
               string lineName = "SLLine_" + IntegerToString((int)dt);
               if(ObjectCreate(0, lineName, OBJ_TREND, 0, barTime, slPrice, time2, slPrice))
                 {
                  ObjectSetInteger(0, lineName, OBJPROP_COLOR, clrRed);
                  ObjectSetInteger(0, lineName, OBJPROP_STYLE, STYLE_DASH);
                  ObjectSetInteger(0, lineName, OBJPROP_WIDTH, 1);
                  ObjectSetInteger(0, lineName, OBJPROP_RAY_LEFT,  false);
                  ObjectSetInteger(0, lineName, OBJPROP_RAY_RIGHT, false);
                  ObjectSetInteger(0, lineName, OBJPROP_BACK, true);
                 }
              }
           }
        }
     }

   FileClose(fileHandle);
   ChartRedraw(0);
   PrintFormat("CSV rows=%d  in window=%d  Buy=%d  Sell=%d",
               totalRows, inWindow, buyCount, sellCount);
  }
