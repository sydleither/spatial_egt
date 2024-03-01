//Based on "Spatial structure impacts adaptive therapy by shaping intra-tumoral competition"
package SpatialEGT;

import HAL.Gui.GridWindow;
import HAL.Rand;

public class SpatialEGT {
    public static void main(String[] args) {
        int x = 100, y = 100;
        int visScale = 4, msPause = 0;
        double divRateS = 0.27;
        double divRateR = 0.75*divRateS;
        double deathRate = 0.25;
        double drugKillRate = 0.075;
        int numCells = (int)((x*y)*0.5);
        double proportionResistant = 0.05;
        GridWindow win = new GridWindow("2D adaptive vs continuous therapy", x*2, y, visScale);
        
        Model2D adaptiveModel2d = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, true);
        adaptiveModel2d.InitTumorRandom(numCells, proportionResistant);

        Model2D continuousModel2d = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, false);
        continuousModel2d.InitTumorRandom(numCells, proportionResistant);

        for (int tick = 0; tick <= 100; tick++) {
            win.TickPause(msPause);
            adaptiveModel2d.ModelStep();
            adaptiveModel2d.DrawModel(win, 0);
            continuousModel2d.ModelStep();
            continuousModel2d.DrawModel(win, 1);
            if (tick % 10 == 0) {
                win.ToPNG("SpatialEGT_2D_" + tick + ".png");
            }
        }
        win.Close();
    }
}
