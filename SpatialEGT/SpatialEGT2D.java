package SpatialEGT;

import java.lang.Math;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import HAL.Gui.GridWindow;
import HAL.Tools.FileIO;
import HAL.Rand;
import HAL.Util;

public class SpatialEGT2D {
    public static HashMap GetPairCorrelation(Model2D model) {
        List<Cell2D> cells = new ArrayList<Cell2D>();
        for (Cell2D cell : model) {
            cells.add(cell);
        }

        HashMap<Integer,Integer> ssPairCounts = new HashMap<Integer,Integer>();
        HashMap<Integer,Integer> srPairCounts = new HashMap<Integer,Integer>();
        HashMap<Integer,Integer> rrPairCounts = new HashMap<Integer,Integer>();
        for (int a = 0; a < cells.size(); a++) {
            for (int b = a+1; b < cells.size(); b++) {
                Cell2D cellA = cells.get(a);
                Cell2D cellB = cells.get(b);
                int cellAtype = cellA.type;
                int cellBtype = cellB.type;
                int dist = (int) Math.round(Util.Dist(cellA.Xsq(), cellA.Ysq(), cellB.Xsq(), cellB.Ysq()));
                if (cellAtype == 0 && cellBtype == 0) {
                    if (ssPairCounts.containsKey(dist)) {
                        ssPairCounts.put(dist, ssPairCounts.get(dist)+1);
                    }
                    else {
                        ssPairCounts.put(dist, 1);
                    }
                }
                if (cellAtype == 1 && cellBtype == 1) {
                    if (rrPairCounts.containsKey(dist)) {
                        rrPairCounts.put(dist, rrPairCounts.get(dist)+1);
                    }
                    else {
                        rrPairCounts.put(dist, 1);
                    }
                }
                else {
                    if (srPairCounts.containsKey(dist)) {
                        srPairCounts.put(dist, srPairCounts.get(dist)+1);
                    }
                    else {
                        srPairCounts.put(dist, 1);
                    }
                }
            }
        }

        HashMap<String, HashMap<Integer,Integer>> returnList = new HashMap<String, HashMap<Integer,Integer>>();
        returnList.put("SS", ssPairCounts);
        returnList.put("SR", srPairCounts);
        returnList.put("RR", rrPairCounts);
        return returnList;
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

    public SpatialEGT2D(String saveLoc, Map<String, Object> params, long seed) {
        // turn parameters json into variables
        int runNull = (int) params.get("null");
        int runContinuous = (int) params.get("continuous");
        int runAdaptive = (int) params.get("adaptive");
        int visualizationFrequency = (int) params.get("visualizationFrequency");
        int writePopFrequency = (int) params.get("writePopFrequency");
        int writePcFrequency = (int) params.get("writePcFrequency");
        int numDays = (int) params.get("numDays");
        int x = (int) params.get("x");
        int y = (int) params.get("y");
        int neighborhood = (int) params.get("neighborhoodRadius");
        double deathRate = (double) params.get("deathRate");
        double drugGrowthReduction = (double) params.get("drugGrowthReduction");
        int numCells = (int) params.get("numCells");
        double proportionResistant = (double) params.get("proportionResistant");
        double adaptiveTreatmentThreshold = (double) params.get("adaptiveTreatmentThreshold");
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
        boolean writePop = writePopFrequency != 0;
        boolean writePc = writePcFrequency != 0;
        boolean visualize = visualizationFrequency != 0;
        FileIO popsOut = null;
        if (writePop) {
            popsOut = new FileIO(saveLoc+"populations.csv", "w");
            popsOut.Write("model,time,sensitive,resistant\n");
        }
        FileIO pcOut = null;
        if (writePc) {
            pcOut = new FileIO(saveLoc+"pairCorrelations.csv", "w");
            pcOut.Write("model,pair,time,distance,count\n");
        }
        GridWindow win = null;
        if (visualize) {
            win = new GridWindow("SpatialEGT", x, y, 4);
        }
        
        // run models
        for (Map.Entry<String,Model2D> modelEntry : models.entrySet()) {
            String modelName = modelEntry.getKey();
            Model2D model = modelEntry.getValue();
            model.InitTumorRandom(numCells, proportionResistant);

            for (int tick = 0; tick <= numDays; tick++) {
                if (writePop) {
                    if (tick % writePopFrequency == 0) {
                        int[] pop = GetPopulationSize(model);
                        popsOut.Write(modelName+","+tick+","+pop[0]+","+pop[1]+"\n");
                    }
                }
                if (writePc) {
                    if (tick % writePcFrequency == 0) {
                        HashMap<String, HashMap<Integer,Integer>> pairCountsList = GetPairCorrelation(model);
                        for (Map.Entry<String, HashMap<Integer,Integer>> pairCountsEntry : pairCountsList.entrySet()) {
                            String pcName = pairCountsEntry.getKey();
                            HashMap<Integer,Integer> pairCounts = pairCountsEntry.getValue();
                            for (Map.Entry<Integer,Integer> pcEntry : pairCounts.entrySet()) {
                                int dist = pcEntry.getKey();
                                int count = pcEntry.getValue();
                                pcOut.Write(modelName+","+pcName+","+tick+","+dist+","+count+"\n");
                            }
                        }
                    }
                }
                if (visualize) {
                    model.DrawModel(win, 0);
                    if (tick % visualizationFrequency == 0) {
                        win.ToPNG(saveLoc+modelName+tick+".png");
                    }
                }
                model.ModelStep();
            }
        }

        // close output files
        if (visualize) {
            win.Close();
        }
        if (writePop) {
            popsOut.Close();
        }
        if (writePc) {
            pcOut.Close();
        }
    }
}
