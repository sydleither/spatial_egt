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
    public static List<List<Integer>> GetModelCoords(Model3D model) {
        List<Integer> cellTypes = new ArrayList<Integer>();
        List<Integer> xCoords = new ArrayList<Integer>();
        List<Integer> yCoords = new ArrayList<Integer>();
        List<Integer> zCoords = new ArrayList<Integer>();
        for (Cell3D cell: model) {
            cellTypes.add(cell.type);
            xCoords.add(cell.Xsq());
            yCoords.add(cell.Ysq());
            zCoords.add(cell.Zsq());
        }
        List<List<Integer>> returnList = new ArrayList<List<Integer>>();
        returnList.add(cellTypes);
        returnList.add(xCoords);
        returnList.add(yCoords);
        returnList.add(zCoords);
        return returnList;
    }

    public SpatialEGT3D(String saveLoc, Map<String, Object> params, long seed) {
        // turn parameters json into variables
        int runNull = (int) params.get("null");
        int runContinuous = (int) params.get("continuous");
        int runAdaptive = (int) params.get("adaptive");
        int writeModelFrequency = (int) params.get("writeModelFrequency");
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
        int totalCells = x*y;
        int z = (int) Math.cbrt(totalCells);

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
        boolean writeModel = writeModelFrequency != 0;
        FileIO modelOut = null;
        if (writeModel) {
            modelOut = new FileIO(saveLoc+"coords.csv", "w");
            modelOut.Write("model,time,type,x,y,z\n");
        }
        
        // run models
        for (Map.Entry<String,Model3D> modelEntry : models.entrySet()) {
            String modelName = modelEntry.getKey();
            Model3D model = modelEntry.getValue();
            model.InitTumorRandom(numCells, proportionResistant);

            for (int tick = 0; tick <= numTicks; tick++) {
                if (writeModel) {
                    if ((tick % writeModelFrequency == 0) && (tick > 0)) {
                        List<List<Integer>> coordLists = GetModelCoords(model);
                        List<Integer> cellTypes = coordLists.get(0);
                        List<Integer> xCoords = coordLists.get(1);
                        List<Integer> yCoords = coordLists.get(2);
                        List<Integer> zCoords = coordLists.get(3);
                        for (int i = 0; i < cellTypes.size(); i++) {
                            modelOut.Write(modelName+","+tick+","+cellTypes.get(i)+","+xCoords.get(i)+","+yCoords.get(i)+","+zCoords.get(i)+"\n");
                        }
                    }
                }
                model.ModelStep();
            }
        }

        // close output files
        if (writeModel) {
            modelOut.Close();
        }
    }
}
