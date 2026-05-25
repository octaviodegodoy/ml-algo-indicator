//+------------------------------------------------------------------+
//| Indicator to plot ML signals from CSV for WIN futures           |
//| Place win_ml_signals.csv in MQL5\Files folder                   |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

input string SignalFile         = "win_ml_signals.csv";
input int    LookbackDays       = 1;     // how many days back to plot signals
input int    RefreshSeconds     = 15;    // how often to re-read the CSV
input bool   OnlyTransitions    = true;  // true = only plot 0->1 buys / 1->0 sells (recommended for M5)
input double FilterMinProb      = 0.52;  // hide BUY arrows whose bar prob < this (match PROB_THRESHOLD in config.py)
input double FilterMinPrecision = 0.52;  // CSV col 'Precision' stores ROC-AUC: hide signals when AUC < this (match MIN_AUC in config.py)
input double FilterMinEdge      = -99.0; // CSV col 'Edge' stores AUC-0.5; disabled (-99) — FilterMinPrecision already covers this

// ── Opening Range Breakout (ORB) display ──────────────────────────────────
// Python writes win_orb.csv each cycle with the first-3-bar range for today.
input bool   ShowORB            = true;             // draw ORB High / Mid / Low lines
input string ORBFile            = "win_orb.csv";    // must match orb_path(slug) in config.py
input color  ORBHighColor       = clrDodgerBlue;    // ORB high breakout line colour
input color  ORBLowColor        = clrOrangeRed;     // ORB low breakdown line colour
input color  ORBMidColor        = clrDimGray;       // midpoint line colour
input int    ORBLineWidth       = 2;                // line thickness

// ── Opening Gap Fill display ──────────────────────────────────────────────
// Python writes win_gap.csv each cycle with prev_close and today_open.
input bool   ShowGap            = true;             // draw PrevClose and TodayOpen lines
input string GapFile            = "win_gap.csv";    // must match gap_path(slug) in config.py
input color  GapPrevCloseColor  = clrGold;          // previous close (gap fill target) line colour
input color  GapOpenColor       = clrMagenta;       // today's open (gap origin) line colour
input int    GapLineWidth       = 2;                // line thickness

//---
int OnInit()
  {
   EventSetTimer(RefreshSeconds);
   PlotSignals(); // first attempt immediately
   PlotORB();
   PlotGap();
   return INIT_SUCCEEDED;
  }

void OnTimer()
  {
   PlotSignals();
   PlotORB();
   PlotGap();
  }

int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[], const double &close[],
                const long &tick_volume[], const long &volume[], const int &spread[])
  {
   if(prev_calculated == 0)
     {
      PlotSignals();
      PlotORB();
      PlotGap();
     }
   return rates_total;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, "BuySignal_");
   ObjectsDeleteAll(0, "SellSignal_");
   ObjectsDeleteAll(0, "SL_");
   ObjectsDeleteAll(0, "SLLine_");
   ObjectsDeleteAll(0, "ORB_");
   ObjectsDeleteAll(0, "GAP_");
  }

//---
// ══ Opening Range Breakout ════════════════════════════════════════════════════
void PlotORB()
  {
   if(!ShowORB) return;

   int fh = FileOpen(ORBFile, FILE_READ|FILE_TXT|FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      // File not written yet (Python hasn't run or ORB not formed).
      // Do NOT delete existing lines — keep whatever was last drawn.
      return;
     }

   // Skip header line
   if(!FileIsEnding(fh)) FileReadString(fh);
   if(FileIsEnding(fh))  { FileClose(fh); return; }

   string line = FileReadString(fh);
   FileClose(fh);
   if(line == "") return;

   string parts[];
   if(StringSplit(line, ',', parts) < 2) return;

   double orb_high = StringToDouble(parts[0]);
   double orb_low  = StringToDouble(parts[1]);
   double orb_mid  = (ArraySize(parts) >= 3) ? StringToDouble(parts[2])
                                             : (orb_high + orb_low) / 2.0;
   if(orb_high <= 0 || orb_low <= 0 || orb_high <= orb_low) return;

   // Data is valid — now safe to redraw
   ObjectsDeleteAll(0, "ORB_");

   // ── ORB High ──────────────────────────────────────────────────────────────
   if(ObjectCreate(0, "ORB_High", OBJ_HLINE, 0, 0, orb_high))
     {
      ObjectSetInteger(0, "ORB_High", OBJPROP_COLOR, ORBHighColor);
      ObjectSetInteger(0, "ORB_High", OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, "ORB_High", OBJPROP_WIDTH, ORBLineWidth);
      ObjectSetString( 0, "ORB_High", OBJPROP_TEXT,
                       "ORB High  " + DoubleToString(orb_high, 0));
      ObjectSetInteger(0, "ORB_High", OBJPROP_BACK, true);
      ObjectSetInteger(0, "ORB_High", OBJPROP_SELECTABLE, false);
     }

   // ── ORB Low ───────────────────────────────────────────────────────────────
   if(ObjectCreate(0, "ORB_Low", OBJ_HLINE, 0, 0, orb_low))
     {
      ObjectSetInteger(0, "ORB_Low", OBJPROP_COLOR, ORBLowColor);
      ObjectSetInteger(0, "ORB_Low", OBJPROP_STYLE, STYLE_DASH);
      ObjectSetInteger(0, "ORB_Low", OBJPROP_WIDTH, ORBLineWidth);
      ObjectSetString( 0, "ORB_Low", OBJPROP_TEXT,
                       "ORB Low  " + DoubleToString(orb_low, 0));
      ObjectSetInteger(0, "ORB_Low", OBJPROP_BACK, true);
      ObjectSetInteger(0, "ORB_Low", OBJPROP_SELECTABLE, false);
     }

   // ── ORB Mid (50% equilibrium level) ──────────────────────────────────────
   if(ObjectCreate(0, "ORB_Mid", OBJ_HLINE, 0, 0, orb_mid))
     {
      ObjectSetInteger(0, "ORB_Mid", OBJPROP_COLOR, ORBMidColor);
      ObjectSetInteger(0, "ORB_Mid", OBJPROP_STYLE, STYLE_DOT);
      ObjectSetInteger(0, "ORB_Mid", OBJPROP_WIDTH, 1);
      ObjectSetString( 0, "ORB_Mid", OBJPROP_TEXT,
                       "ORB Mid  " + DoubleToString(orb_mid, 0));
      ObjectSetInteger(0, "ORB_Mid", OBJPROP_BACK, true);
      ObjectSetInteger(0, "ORB_Mid", OBJPROP_SELECTABLE, false);
     }

   // ── ORB Breakout arrows ───────────────────────────────────────────────────
   // Scan today's bars (after 09:15 BRT, once the 3-bar ORB is complete).
   // Draw ONE arrow at the first bar whose close breaks above orb_high (BUY)
   // or below orb_low (SELL).  Both directions are tracked independently.
   {
      string      sym = _Symbol;
      ENUM_TIMEFRAMES tf = _Period;

      // Build today's midnight timestamp (broker clock = BRT)
      MqlDateTime nd;
      TimeToStruct(TimeCurrent(), nd);
      nd.hour = 0; nd.min = 0; nd.sec = 0;
      datetime today_start = StructToTime(nd);

      // Use a fixed 300-bar window so we never miss today's session even when
      // iBarShift(midnight) returns -1 (no bar exists at midnight for intraday).
      bool buy_drawn = false, sell_drawn = false;
      int  scan_start = MathMin(300, Bars(sym, tf) - 1);

      // Loop chronologically: high index = older, low index = newer
      for(int i = scan_start; i >= 0; i--)
        {
         datetime bt = iTime(sym, tf, i);
         if(bt < today_start) continue;            // pre-today bar, skip

         MqlDateTime bdt;
         TimeToStruct(bt, bdt);
         // Ignore bars inside the 3-bar ORB formation window (09:00–09:14)
         if(bdt.hour == 9 && bdt.min < 15) continue;

         double c = iClose(sym, tf, i);

         // ── First upside breakout ──────────────────────────────────────────
         if(!buy_drawn && c > orb_high)
           {
            double ap = iLow(sym, tf, i) * 0.9994;
            if(ObjectCreate(0, "ORB_BuyBreak", OBJ_ARROW, 0, bt, ap))
              {
               ObjectSetInteger(0, "ORB_BuyBreak", OBJPROP_ARROWCODE, 233);
               ObjectSetInteger(0, "ORB_BuyBreak", OBJPROP_COLOR,     ORBHighColor);
               ObjectSetInteger(0, "ORB_BuyBreak", OBJPROP_WIDTH,     3);
               ObjectSetInteger(0, "ORB_BuyBreak", OBJPROP_BACK,      false);
              }
            ObjectCreate(0, "ORB_BuyLabel", OBJ_TEXT, 0, bt, ap * 0.9990);
            ObjectSetString( 0, "ORB_BuyLabel", OBJPROP_TEXT,     "ORB+");
            ObjectSetInteger(0, "ORB_BuyLabel", OBJPROP_COLOR,    ORBHighColor);
            ObjectSetInteger(0, "ORB_BuyLabel", OBJPROP_FONTSIZE, 9);
            ObjectSetInteger(0, "ORB_BuyLabel", OBJPROP_ANCHOR,   ANCHOR_TOP);
            buy_drawn = true;
           }

         // ── First downside breakout ────────────────────────────────────────
         if(!sell_drawn && c < orb_low)
           {
            double ap = iHigh(sym, tf, i) * 1.0006;
            if(ObjectCreate(0, "ORB_SellBreak", OBJ_ARROW, 0, bt, ap))
              {
               ObjectSetInteger(0, "ORB_SellBreak", OBJPROP_ARROWCODE, 234);
               ObjectSetInteger(0, "ORB_SellBreak", OBJPROP_COLOR,     ORBLowColor);
               ObjectSetInteger(0, "ORB_SellBreak", OBJPROP_WIDTH,     3);
               ObjectSetInteger(0, "ORB_SellBreak", OBJPROP_BACK,      false);
              }
            ObjectCreate(0, "ORB_SellLabel", OBJ_TEXT, 0, bt, ap * 1.0010);
            ObjectSetString( 0, "ORB_SellLabel", OBJPROP_TEXT,     "ORB-");
            ObjectSetInteger(0, "ORB_SellLabel", OBJPROP_COLOR,    ORBLowColor);
            ObjectSetInteger(0, "ORB_SellLabel", OBJPROP_FONTSIZE, 9);
            ObjectSetInteger(0, "ORB_SellLabel", OBJPROP_ANCHOR,   ANCHOR_BOTTOM);
            sell_drawn = true;
           }

         if(buy_drawn && sell_drawn) break;
        }
   }

   ChartRedraw(0);
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
      double rowProb      = nParts >= 4 ? StringToDouble(parts[3]) : 1.0;
      double rowPrecision = nParts >= 5 ? StringToDouble(parts[4]) : 1.0;
      double rowEdge      = nParts >= 6 ? StringToDouble(parts[5]) : 1.0;
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

      // ── Quality / probability filter ──────────────────────────────────────
      // Model-level gate: hide everything when precision or edge is too low
      if(rowPrecision < FilterMinPrecision || rowEdge < FilterMinEdge) continue;
      // Per-bar gate: hide BUY arrows whose raw probability is below the threshold
      if(isBuy && rowProb < FilterMinProb) continue;

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

   // Only log when something changes (avoids journal spam on every 15-s refresh)
   static int s_lastTotal = -1, s_lastBuy = -1, s_lastSell = -1;
   if(totalRows != s_lastTotal || buyCount != s_lastBuy || sellCount != s_lastSell)
     {
      PrintFormat("CSV rows=%d  in window=%d  Buy=%d  Sell=%d",
                  totalRows, inWindow, buyCount, sellCount);
      s_lastTotal = totalRows;
      s_lastBuy   = buyCount;
      s_lastSell  = sellCount;
     }
  }

//---
// ══ Opening Gap Fill ══════════════════════════════════════════════════════════
void PlotGap()
  {
   if(!ShowGap) return;

   int fh = FileOpen(GapFile, FILE_READ|FILE_TXT|FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      // File not written yet (Python hasn't run or no gap today).
      // Keep whatever lines were last drawn.
      return;
     }

   // Skip header line
   if(!FileIsEnding(fh)) FileReadString(fh);
   if(FileIsEnding(fh))  { FileClose(fh); return; }

   string line = FileReadString(fh);
   FileClose(fh);
   if(line == "") return;

   string parts[];
   if(StringSplit(line, ',', parts) < 3) return;

   double prev_close  = StringToDouble(parts[0]);
   double today_open  = StringToDouble(parts[1]);
   double gap_size    = StringToDouble(parts[2]);
   int    gap_dir     = (ArraySize(parts) >= 4) ? (int)StringToInteger(parts[3]) : 0;

   if(prev_close <= 0 || today_open <= 0 || gap_size <= 0) return;

   // Data is valid — redraw
   ObjectsDeleteAll(0, "GAP_");

   // ── Previous Close (gap fill target) ─────────────────────────────────────
   string label_pc = StringFormat("GAP_PrevClose  %.0f  (gap %s %.0f pts)",
                                  prev_close,
                                  gap_dir > 0 ? "▲" : "▼",
                                  gap_size);
   if(ObjectCreate(0, "GAP_PrevClose", OBJ_HLINE, 0, 0, prev_close))
     {
      ObjectSetInteger(0, "GAP_PrevClose", OBJPROP_COLOR,     GapPrevCloseColor);
      ObjectSetInteger(0, "GAP_PrevClose", OBJPROP_STYLE,     STYLE_SOLID);
      ObjectSetInteger(0, "GAP_PrevClose", OBJPROP_WIDTH,     GapLineWidth);
      ObjectSetString( 0, "GAP_PrevClose", OBJPROP_TEXT,      label_pc);
      ObjectSetInteger(0, "GAP_PrevClose", OBJPROP_BACK,      true);
      ObjectSetInteger(0, "GAP_PrevClose", OBJPROP_SELECTABLE,false);
     }

   // ── Today's Open (gap origin) ─────────────────────────────────────────────
   if(ObjectCreate(0, "GAP_Open", OBJ_HLINE, 0, 0, today_open))
     {
      ObjectSetInteger(0, "GAP_Open", OBJPROP_COLOR,     GapOpenColor);
      ObjectSetInteger(0, "GAP_Open", OBJPROP_STYLE,     STYLE_DOT);
      ObjectSetInteger(0, "GAP_Open", OBJPROP_WIDTH,     1);
      ObjectSetString( 0, "GAP_Open", OBJPROP_TEXT,
                       "GAP Open  " + DoubleToString(today_open, 0));
      ObjectSetInteger(0, "GAP_Open", OBJPROP_BACK,      true);
      ObjectSetInteger(0, "GAP_Open", OBJPROP_SELECTABLE,false);
     }

   ChartRedraw(0);
  }
