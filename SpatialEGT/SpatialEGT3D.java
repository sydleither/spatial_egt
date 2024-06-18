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
    public static List GetPairCorrelation(Model3D model, int maxDistance, int maxEuclideanDistance) {
        List<Cell3D> cells = new ArrayList<Cell3D>();
        for (Cell3D cell : model) {
            cells.add(cell);
        }

        Map<Integer, Integer[]> xRows = new HashMap<>();
        Map<Integer, Integer[]> yRows = new HashMap<>();
        Map<Integer, Integer[]> zRows = new HashMap<>();
        Map<Integer, Integer[]> euclideanRows = new HashMap<>();
        for (int i = 0; i < maxDistance; i++) {
            Integer xStartingList[] = {0, 0, 0};
            xRows.put(i, xStartingList);
            Integer yStartingList[] = {0, 0, 0};
            yRows.put(i, yStartingList);
            Integer zStartingList[] = {0, 0, 0};
            zRows.put(i, zStartingList);
            Integer euclideanStartingList[] = {0, 0, 0};
            euclideanRows.put(i, euclideanStartingList);
        }
        for (int i = maxDistance; i < maxEuclideanDistance; i++) {
            Integer euclideanStartingList[] = {0, 0, 0};
            euclideanRows.put(i, euclideanStartingList);
        }

        for (int a = 0; a < cells.size(); a++) {
            for (int b = a+1; b < cells.size(); b++) {
                Cell3D cellA = cells.get(a);
                Cell3D cellB = cells.get(b);
                int cellAtype = cellA.type;
                int cellBtype = cellB.type;
                int xDistance = Math.abs(cellA.Xsq() - cellB.Xsq());
                int yDistance = Math.abs(cellA.Ysq() - cellB.Ysq());
                int zDistance = Math.abs(cellA.Zsq() - cellB.Zsq());
                int euclideanDistance = (int) Math.round(Util.Dist(cellA.Xsq(), cellA.Ysq(), cellA.Zsq(), cellB.Xsq(), cellB.Ysq(), cellB.Zsq()));
                
                int pairType;
                if (cellAtype == 0 && cellBtype == 0)
                    pairType = 0;
                else if (cellAtype == 1 && cellBtype == 1)
                    pairType = 2;
                else
                    pairType = 1;

                xRows.get(xDistance)[pairType] = xRows.get(xDistance)[pairType]+1;
                yRows.get(yDistance)[pairType] = yRows.get(yDistance)[pairType]+1;
                zRows.get(zDistance)[pairType] = zRows.get(zDistance)[pairType]+1;
                euclideanRows.get(euclideanDistance)[pairType] = euclideanRows.get(euclideanDistance)[pairType]+1;
            }
        }

        List<Map<Integer, Integer[]>> returnList = new ArrayList<>();
        returnList.add(xRows);
        returnList.add(yRows);
        returnList.add(zRows);
        returnList.add(euclideanRows);
        return returnList;
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
        int numTicks = (int) params.get("numTicks");
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
        int totalCells = x*y;
        int z = (int) Math.cbrt(totalCells);

        // calculations for pair correlation
        int maxDistance = z;
        int maxEuclideanDistance = (int) Math.round(Math.sqrt(3*Math.pow(maxDistance, 2)));

        // initialize with specified models
        HashMap<String,Model3D> models = new HashMap<String,Model3D>();
        if (runNull == 1) {
            Model3D nullModel = new Model3D(z, z, z, new Rand(seed), neighborhood, deathRate, 0.0, false, 0.0, payoff);
            models.put("nodrug", nullModel);
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
            pcOut.Write("model,time,pair,measure,distance,count\n");
        }
        
        // run models
        for (Map.Entry<String,Model3D> modelEntry : models.entrySet()) {
            String modelName = modelEntry.getKey();
            Model3D model = modelEntry.getValue();
            model.InitTumorRandom(numCells, proportionResistant);

            for (int tick = 0; tick <= numTicks; tick++) {
                if (writePop) {
                    if (tick % writePopFrequency == 0) {
                        int[] pop = GetPopulationSize(model);
                        popsOut.Write(modelName+","+tick+","+pop[0]+","+pop[1]+"\n");
                    }
                }
                if (writePc) {
                    if (tick % writePcFrequency == 0) {
                        List<Map<Integer, Integer[]>> pairCountsList = GetPairCorrelation(model, maxDistance, maxEuclideanDistance);
                        Map<Integer, Integer[]> xRows = pairCountsList.get(0);
                        Map<Integer, Integer[]> yRows = pairCountsList.get(1);
                        Map<Integer, Integer[]> zRows = pairCountsList.get(2);
                        Map<Integer, Integer[]> euclideanRows = pairCountsList.get(3);

                        String pairTypes[] = {"SS", "SR", "RR"};
                        for (int dist = 0; dist < maxDistance; dist++) {
                            for (int i = 0; i < 3; i++) {
                                pcOut.Write(modelName+","+tick+","+pairTypes[i]+",x,"+dist+","+xRows.get(dist)[i]+"\n");
                                pcOut.Write(modelName+","+tick+","+pairTypes[i]+",y,"+dist+","+yRows.get(dist)[i]+"\n");
                                pcOut.Write(modelName+","+tick+","+pairTypes[i]+",z,"+dist+","+zRows.get(dist)[i]+"\n");
                                pcOut.Write(modelName+","+tick+","+pairTypes[i]+",euclidean,"+dist+","+euclideanRows.get(dist)[i]+"\n");
                            }
                        }
                        for (int dist = maxDistance; dist < maxEuclideanDistance; dist++) {
                            for (int i = 0; i < 3; i++) {
                                pcOut.Write(modelName+","+tick+","+pairTypes[i]+",euclidean,"+dist+","+euclideanRows.get(dist)[i]+"\n");
                            }
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
