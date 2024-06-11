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

public class SpatialEGT3D {
    //split this between resistant and sensitive
    public static HashMap<Integer,Integer> GetPairCorrelation(Model3D model) {
        List<Cell3D> cells = new ArrayList<Cell3D>();
        for (Cell3D cell : model) {
            cells.add(cell);
        }

        HashMap<Integer,Integer> pairCounts = new HashMap<Integer,Integer>();
        for (int a = 0; a < cells.size(); a++) {
            for (int b = a+1; b < cells.size(); b++) {
                Cell3D cellA = cells.get(a);
                Cell3D cellB = cells.get(b);
                int dist = (int) Math.round(Util.Dist(cellA.Xsq(), cellA.Ysq(), cellB.Xsq(), cellB.Ysq()));
                if (pairCounts.containsKey(dist)) {
                    pairCounts.put(dist, pairCounts.get(dist)+1);
                }
                else {
                    pairCounts.put(dist, 1);
                }
            }
        }
        return pairCounts;
    }

    public static int[] GetPopulationSize(Model3D model) {
        int numResistant = 0;
        int numSensitive = 0;
        for (Cell3D cell : model) {
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

    public SpatialEGT3D(String saveLoc, Map<String, Object> params, long seed) {
        // turn parameters json into variables
        int runNull = (int) params.get("null");
        int runContinuous = (int) params.get("continuous");
        int runAdaptive = (int) params.get("adaptive");
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
        int totalCells = x*y;
        int z = (int) Math.cbrt(totalCells);
        HashMap<String,Model3D> models = new HashMap<String,Model3D>();
        if (runNull == 1) {
            Model3D nullModel = new Model3D(z, z, z, new Rand(seed), neighborhood, deathRate, 0.0, false, 0.0, payoff);
            models.put("null", nullModel);
        }
        if (runContinuous == 1) {
            Model3D continuousModel = new Model3D(z, z, z, new Rand(seed), neighborhood, deathRate, drugGrowthReduction, false, 0.0, payoff);
            models.put("continuous", continuousModel);
        }
        if (runAdaptive == 1) {
            Model3D adaptiveModel = new Model3D(z, z, z, new Rand(seed), neighborhood, deathRate, drugGrowthReduction, true, adaptiveTreatmentThreshold, payoff);
            models.put("adaptive", adaptiveModel);
        }

        // check what to run and initialize output
        boolean writePop = writePopFrequency != 0;
        boolean writePc = writePcFrequency != 0;
        FileIO popsOut = null;
        if (writePop) {
            popsOut = new FileIO(saveLoc+"populations.csv", "w");
            popsOut.Write("model,time,sensitive,resistant\n");
        }
        FileIO pcOut = null;
        if (writePc) {
            pcOut = new FileIO(saveLoc+"pairCorrelations.csv", "w");
            pcOut.Write("model,time,distance,count\n");
        }
        
        // run models
        for (Map.Entry<String,Model3D> modelEntry : models.entrySet()) {
            String modelName = modelEntry.getKey();
            Model3D model = modelEntry.getValue();
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
                        HashMap<Integer,Integer> pairCounts = GetPairCorrelation(model);
                        for (Map.Entry<Integer,Integer> pcEntry : pairCounts.entrySet()) {
                            int dist = pcEntry.getKey();
                            int count = pcEntry.getValue();
                            pcOut.Write(modelName+","+tick+","+dist+","+count+"\n");
                        }
                    }
                }
                model.ModelStep();
            }
        }

        // close output files
        if (writePop) {
            popsOut.Close();
        }
        if (writePc) {
            pcOut.Close();
        }
    }
}
