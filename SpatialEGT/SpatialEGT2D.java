package SpatialEGT;

import java.util.Arrays;
import java.lang.Math;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import HAL.Gui.GifMaker;
import HAL.Gui.GridWindow;
import HAL.Tools.FileIO;
import HAL.Rand;
import HAL.Util;

public class SpatialEGT2D {
    public static void SaveModelState(FileIO modelOut, Model2D model, int x, int y) {
        String row;
        for (int i = 0; i < x; i++) {
            row = "";
            for (int j = 0; j < y; j++) {
                Cell2D cell = model.GetAgent(i,j);
                String state;
                if (cell == null) {
                    state = "*";
                }
                else if (cell.type == 0) {
                    state = "S";
                }
                else if (cell.type == 1) {
                    state = "R";
                }
                else {
                    state = "?";
                }
                row += state;
            }
            modelOut.Write(row+"\n");
        }
        modelOut.Close();
    }

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

    public SpatialEGT2D(String saveLoc, Map<String, Object> params, long seed, int visualizationFrequency) {
        // turn parameters json into variables
        int runNull = (int) params.get("null");
        int runContinuous = (int) params.get("continuous");
        int runAdaptive = (int) params.get("adaptive");
        int writeModelFrequency = (int) params.get("writeModelFrequency");
        int writePopFrequency = (int) params.get("writePopFrequency");
        int numTicks = (int) params.get("numTicks");
        int x = (int) params.get("x");
        int y = (int) params.get("y");
        int neighborhood = (int) params.get("neighborhoodRadius");
        double deathRate = (double) params.get("deathRate");
        double drugGrowthReduction = (double) params.get("drugGrowthReduction");
        int numCells = (int) params.get("numCells");
        double proportionResistant = (double) params.get("proportionResistant");
        double adaptiveTreatmentThreshold = (double) params.get("adaptiveTreatmentThreshold");
        int initialTumor = (int) params.get("initialTumor");
        int toyGap = (int) params.get("toyGap");
        double[][] payoff = new double[2][2];
        payoff[0][0] = (double) params.get("A");
        payoff[0][1] = (double) params.get("B");
        payoff[1][0] = (double) params.get("C");
        payoff[1][1] = (double) params.get("D");

        // initialize with specified models
        HashMap<String,Model2D> models = new HashMap<String,Model2D>();
        if (runNull == 1) {
            Model2D nullModel = new Model2D(x, y, new Rand(seed), neighborhood, deathRate, 0.0, false, 0.0, payoff);
            models.put("nodrug", nullModel);
        }
        if (runContinuous == 1) {
            Model2D continuousModel = new Model2D(x, y, new Rand(seed), neighborhood, deathRate, drugGrowthReduction, false, 0.0, payoff);
            models.put("continuous", continuousModel);
        }
        if (runAdaptive == 1) {
            Model2D adaptiveModel = new Model2D(x, y, new Rand(seed), neighborhood, deathRate, drugGrowthReduction, true, adaptiveTreatmentThreshold, payoff);
            models.put("adaptive", adaptiveModel);
        }

        // check what to run and initialize output
        boolean writeModel = writeModelFrequency != 0;
        boolean writePop = writePopFrequency != 0;
        boolean visualize = visualizationFrequency != 0;

        GridWindow win = null;
        GifMaker gifWin = null;
        if (visualize) {
            win = new GridWindow("SpatialEGT", x, y, 4);
            gifWin = new GifMaker(saveLoc+"growth.gif", 0, false);
            writePop = false;
            writeModel = false;
        }
        FileIO popsOut = null;
        if (writePop) {
            popsOut = new FileIO(saveLoc+"populations.csv", "w");
            popsOut.Write("model,time,sensitive,resistant\n");
        }
        
        // run models
        for (Map.Entry<String,Model2D> modelEntry : models.entrySet()) {
            String modelName = modelEntry.getKey();
            Model2D model = modelEntry.getValue();
            if (initialTumor == 1)
                model.InitTumorLinear(proportionResistant, toyGap);
            else if (initialTumor == 2)
                model.InitTumorConvex(numCells, proportionResistant);
            else if (initialTumor == 3)
                model.InitTumorConcave(numCells, proportionResistant);
            else if (initialTumor == 4)
                model.InitTumorCircle(proportionResistant, toyGap);
            else
                model.InitTumorRandom(numCells, proportionResistant);

            for (int tick = 0; tick <= numTicks; tick++) {
                if (writePop) {
                    if (tick % writePopFrequency == 0) {
                        int[] pop = GetPopulationSize(model);
                        popsOut.Write(modelName+","+tick+","+pop[0]+","+pop[1]+"\n");
                    }
                }
                if (writeModel) {
                    if ((tick % writeModelFrequency == 0) && (tick > 0)) {
                        FileIO modelOut = new FileIO(saveLoc+"model"+tick+".csv", "w");
                        SaveModelState(modelOut, model, x, y);
                    }
                }
                if (visualize) {
                    model.DrawModel(win, 0);
                    if (tick % visualizationFrequency == 0) {
                        win.ToPNG(saveLoc+modelName+tick+".png");
                    }
                    gifWin.AddFrame(win);
                }
                model.ModelStep();
            }
        }

        // close output files
        if (visualize) {
            win.Close();
            gifWin.Close();
        }
        if (writePop) {
            popsOut.Close();
        }
    }
}
