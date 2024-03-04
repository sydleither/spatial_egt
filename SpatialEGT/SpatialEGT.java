//Based on "Spatial structure impacts adaptive therapy by shaping intra-tumoral competition"
package SpatialEGT;

import HAL.Gui.GridWindow;
import HAL.Tools.FileIO;
import HAL.Rand;

public class SpatialEGT {
    public static int[] GetPopulationSize(Model2D model) {
        int numResistant = 0;
        int numSensitive = 0;
        for (Cell2D cell : model) {
            if (cell.type == 0) {
                numSensitive += 1;
            }
            else {
                numResistant += 1;
            }
        }
        int[] popSize = new int[]{numSensitive, numResistant};
        return popSize;
    }

    public static void main(String[] args) {
        int x = 100, y = 100;
        int visScale = 4, msPause = 0;
        double divRateS = 0.027;
        double divRateR = 0.6*divRateS;
        double deathRate = 0.5*divRateS;
        double drugKillRate = 0.75;
        int numCells = (int)((x*y)*0.25);
        double proportionResistant = 0.05;
        GridWindow win = new GridWindow("2D adaptive vs continuous therapy", x*2, y, visScale);
        FileIO popsOut = new FileIO("SpatialEGTPopulations.csv", "w");
        popsOut.Write("adaptive_sensitive, adaptive_resistant, continuous_sensitive, continuous_resistant\n");

        Model2D adaptiveModel2d = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, true);
        adaptiveModel2d.InitTumorRandom(numCells, proportionResistant);

        Model2D continuousModel2d = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, false);
        continuousModel2d.InitTumorRandom(numCells, proportionResistant);

        for (int tick = 0; tick <= 2000; tick++) {
            win.TickPause(msPause);

            adaptiveModel2d.ModelStep();
            //adaptiveModel2d.DrawModel(win, 0);
            continuousModel2d.ModelStep();
            //continuousModel2d.DrawModel(win, 1);

            if (tick % 10 == 0) {
                int[] adaptivePop = GetPopulationSize(adaptiveModel2d);
                int[] continuousPop = GetPopulationSize(continuousModel2d);
                popsOut.Write(adaptivePop[0] + "," + adaptivePop[1] + "," + continuousPop[0] + "," + continuousPop[1] + "\n");
            }
            if (tick % 250 == 0) {
                win.ToPNG("SpatialEGT_2D_" + tick + ".png");
            }
        }
        
        popsOut.Close();
        win.Close();
    }
}
