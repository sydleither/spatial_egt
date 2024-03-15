//Based on "Spatial structure impacts adaptive therapy by shaping intra-tumoral competition"
package SpatialEGT;

import java.nio.file.Paths;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

import HAL.Gui.GridWindow;
import HAL.Tools.FileIO;
import HAL.Rand;
import HAL.GridsAndAgents.AgentBaseSpatial;

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

    public static int[] GetPopulationSize(Model0D model) {
        int numResistant = 0;
        int numSensitive = 0;
        for (Cell0D cell : model) {
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

    public static void RunModels(String exp_name, String dimension, Model2D adaptiveModel, Model2D continuousModel, GridWindow win, FileIO popsOut, int numDays) {
        for (int tick = 0; tick <= numDays; tick++) {
            adaptiveModel.ModelStep();
            adaptiveModel.DrawModel(win, 0);
            continuousModel.ModelStep();
            continuousModel.DrawModel(win, 1);

            if (tick % 10 == 0) {
                int[] adaptivePop = GetPopulationSize(adaptiveModel);
                int[] continuousPop = GetPopulationSize(continuousModel);
                popsOut.Write(tick + "," + adaptivePop[0] + "," + adaptivePop[1] + "," + continuousPop[0] + "," + continuousPop[1] + "\n");
            }
            if (tick % (int)(numDays/10) == 0) {
                win.ToPNG("output/" + exp_name + "/" + dimension + "model_tick" + tick + ".png");
            }
        }
    }

    public static void RunModels(String exp_name, String dimension, Model0D adaptiveModel, Model0D continuousModel, GridWindow win, FileIO popsOut, int numDays) {
        for (int tick = 0; tick <= numDays; tick++) {
            adaptiveModel.ModelStep();
            adaptiveModel.DrawModel(win, 0);
            continuousModel.ModelStep();
            continuousModel.DrawModel(win, 1);

            if (tick % 10 == 0) {
                int[] adaptivePop = GetPopulationSize(adaptiveModel);
                int[] continuousPop = GetPopulationSize(continuousModel);
                popsOut.Write(tick + "," + adaptivePop[0] + "," + adaptivePop[1] + "," + continuousPop[0] + "," + continuousPop[1] + "\n");
            }
            if (tick % (int)(numDays/10) == 0) {
                win.ToPNG("output/" + exp_name + "/" + dimension + "model_tick" + tick + ".png");
            }
        }
    }

    public static void main(String[] args) {
        String exp_name = args[0];
        String dimension = args[1];
        ObjectMapper mapper = new ObjectMapper();
        Map<String, ?> params;
        try{
            params = mapper.readValue(Paths.get(exp_name+".json").toFile(), Map.class);
        }
        catch (Exception e) {
            return;
        }

        int visScale = 4;
        int numDays = (int) params.get("numDays");
        int x = (int) params.get("x");
        int y = (int) params.get("y");
        double deathRate = (double) params.get("deathRate");
        double drugKillRate = (double) params.get("drugKillRate");
        int numCells = (int) params.get("numCells");
        double proportionResistant = (double) params.get("proportionResistant");
        boolean egt = (boolean) params.get("egt");
        double divRateS = 0;
        double divRateR = 0;
        int[][] payoff = new int[2][2];
        if (egt) {
            payoff[0][0] = (int) params.get("A");
            payoff[0][1] = (int) params.get("B");
            payoff[1][0] = (int) params.get("C");
            payoff[1][1] = (int) params.get("D");
        }
        else {
            divRateS = (double) params.get("divRateS");
            divRateR = (double) params.get("divRateR");
        }

        GridWindow win = new GridWindow(dimension+" adaptive vs continuous therapy", x*2, y, visScale);
        FileIO popsOut = new FileIO("output/"+exp_name+"/"+dimension+"populations.csv", "w");
        popsOut.Write("time,adaptive_sensitive,adaptive_resistant,continuous_sensitive,continuous_resistant\n");

        if (dimension.equals("2D")) {
            Model2D adaptiveModel = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, true, egt, payoff);
            Model2D continuousModel = new Model2D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, false, egt, payoff);
            adaptiveModel.InitTumorRandom(numCells, proportionResistant);
            continuousModel.InitTumorRandom(numCells, proportionResistant);    
            RunModels(exp_name, dimension, adaptiveModel, continuousModel, win, popsOut, numDays);
        }
        else if (dimension.equals("0D")) {
            Model0D adaptiveModel = new Model0D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, true, egt, payoff);
            Model0D continuousModel = new Model0D(x, y, new Rand(), divRateS, divRateR, deathRate, drugKillRate, false, egt, payoff);
            adaptiveModel.InitTumorRandom(numCells, proportionResistant);
            continuousModel.InitTumorRandom(numCells, proportionResistant);    
            RunModels(exp_name, dimension, adaptiveModel, continuousModel, win, popsOut, numDays);
        }

        popsOut.Close();
        win.Close();
    }
}
