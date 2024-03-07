//Based on "Spatial structure impacts adaptive therapy by shaping intra-tumoral competition"
package SpatialEGT;

import java.nio.file.Paths;
import java.util.Map; 

import com.fasterxml.jackson.databind.ObjectMapper;

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
        String exp_name = args[0];
        ObjectMapper mapper = new ObjectMapper();
        Map<String, ?> params;
        try{
            params = mapper.readValue(Paths.get(exp_name+".json").toFile(), Map.class);
        }
        catch (Exception e) {
            return;
        }

        int visScale = 4, msPause = 0;
        int numDays = (int) params.get("numDays");
        int x = (int) params.get("x");
        int y = (int) params.get("y");
        double divRateS = (double) params.get("divRateS");
        double divRateR = (double) params.get("divRateR");
        double deathRate = (double) params.get("deathRate");
        double drugKillRate = (double) params.get("drugKillRate");
        int numCells = (int) params.get("numCells");
        double proportionResistant = (double) params.get("proportionResistant");

        GridWindow win = new GridWindow("2D adaptive vs continuous therapy", x*2, y, visScale);
        FileIO popsOut = new FileIO("output/"+exp_name+"/populations.csv", "w");
        popsOut.Write("time,adaptive_sensitive,adaptive_resistant,continuous_sensitive,continuous_resistant\n");

        Model2D adaptiveModel2d = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, true);
        adaptiveModel2d.InitTumorRandom(numCells, proportionResistant);

        Model2D continuousModel2d = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, false);
        continuousModel2d.InitTumorRandom(numCells, proportionResistant);

        for (int tick = 0; tick <= numDays; tick++) {
            win.TickPause(msPause);

            adaptiveModel2d.ModelStep();
            adaptiveModel2d.DrawModel(win, 0);
            continuousModel2d.ModelStep();
            continuousModel2d.DrawModel(win, 1);

            if (tick % 10 == 0) {
                int[] adaptivePop = GetPopulationSize(adaptiveModel2d);
                int[] continuousPop = GetPopulationSize(continuousModel2d);
                popsOut.Write(tick + "," + adaptivePop[0] + "," + adaptivePop[1] + "," + continuousPop[0] + "," + continuousPop[1] + "\n");
            }
            if (tick % (int)(numDays/10) == 0) {
                win.ToPNG("output/" + exp_name + "/model_tick" + tick + ".png");
            }
        }

        popsOut.Close();
        win.Close();
    }
}
