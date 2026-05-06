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

   PrintFormat("PlotSignals: chart=%s/%s  bars=%d  lastBar=%s  cutoff=%s",
               sym, EnumToString(tf), barsAvail,
               TimeToString(lastBarTime), TimeToString(cutoff));

   if(barsAvail < 10)
     {
      Print("Series not ready — will retry on next tick");
      return;
     }

   ObjectsDeleteAll(0, "BuySignal_");
   ObjectsDeleteAll(0, "SellSignal_");

   int fileHandle = FileOpen(SignalFile, FILE_READ|FILE_CSV|FILE_ANSI, ',');
   if(fileHandle == INVALID_HANDLE)
     {
      Print("Failed to open ", SignalFile, " error=", GetLastError());
      return;
     }

   // Skip BOTH header fields ("Timestamp" and "ML_Signal")
   FileReadString(fileHandle);
   FileReadString(fileHandle);

   int buyCount = 0, sellCount = 0, totalRows = 0, inWindow = 0;
   int prevSignal = -1;

   while(!FileIsEnding(fileHandle))
     {
      string timestamp = FileReadString(fileHandle);
      string sigStr    = FileReadString(fileHandle);
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

      if(isBuy)
        {
         string name = "BuySignal_" + IntegerToString((int)dt);
         if(ObjectCreate(0, name, OBJ_ARROW, 0, barTime, iLow(sym, tf, bar) * 0.9995))
           {
            ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 233);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrLime);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, name, OBJPROP_BACK, false);
            buyCount++;
           }
        }
      else
        {
         string name = "SellSignal_" + IntegerToString((int)dt);
         if(ObjectCreate(0, name, OBJ_ARROW, 0, barTime, iHigh(sym, tf, bar) * 1.0005))
           {
            ObjectSetInteger(0, name, OBJPROP_ARROWCODE, 234);
            ObjectSetInteger(0, name, OBJPROP_COLOR, clrRed);
            ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, name, OBJPROP_BACK, false);
            sellCount++;
           }
        }
     }

   FileClose(fileHandle);
   ChartRedraw(0);
   PrintFormat("CSV rows=%d  in window=%d  Buy=%d  Sell=%d",
               totalRows, inWindow, buyCount, sellCount);
  }
