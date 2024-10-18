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
    public static List<List<Integer>> GetModelCoords(Model2D model) {
        List<Integer> cellTypes = new ArrayList<Integer>();
        List<Integer> xCoords = new ArrayList<Integer>();
        List<Integer> yCoords = new ArrayList<Integer>();
        for (Cell2D cell: model) {
            cellTypes.add(cell.type);
            xCoords.add(cell.Xsq());
            yCoords.add(cell.Ysq());
        }
        List<List<Integer>> returnList = new ArrayList<List<Integer>>();
        returnList.add(cellTypes);
        returnList.add(xCoords);
        returnList.add(yCoords);
        return returnList;
    }

    public static Map<List<Double>, List<Integer>> GetFsList(Model2D model, int maxRadius) {
        ArrayList<List<Object>> fsList = new ArrayList<List<Object>>();
        for (Cell2D cell: model) {
            if (cell.type == 0) {
                continue;
            }
            HashMap<Integer,Double> fsListCell = cell.Fs(maxRadius);
            for (int radius = 1; radius <= maxRadius; radius++) {
                List<Object> listEntry = Arrays.asList(cell.reproduced, (double) radius, fsListCell.get(radius));
                fsList.add(listEntry);
            }
        }

        Map<List<Double>, List<Integer>> fsListBinned = new HashMap<>();
        for (List<Object> listEntry : fsList) {
            int reproduced = (boolean) listEntry.get(0) ? 1 : 0;
            double radius = (double) listEntry.get(1);
            double fs = (double) listEntry.get(2);
            fs = Math.round(fs * Math.pow(10, 2)) / Math.pow(10, 2);
            List<Double> listBinnedKey = Arrays.asList(radius, fs);
            if (fsListBinned.get(listBinnedKey) == null) {
                List<Integer> listBinnedEntry = Arrays.asList(reproduced, 1);
                fsListBinned.put(listBinnedKey, listBinnedEntry);
            }
            else {
                List<Integer> listBinnedEntry = Arrays.asList(fsListBinned.get(listBinnedKey).get(0)+reproduced, fsListBinned.get(listBinnedKey).get(1)+1);
                fsListBinned.put(listBinnedKey, listBinnedEntry);
            }
        }

        return fsListBinned;
    }

    public static Map<List<Double>, List<Integer>> GetFrList(Model2D model, int maxRadius) {
        ArrayList<List<Object>> frList = new ArrayList<List<Object>>();
        for (Cell2D cell: model) {
            if (cell.type == 1) {
                continue;
            }
            HashMap<Integer,Double> frListCell = cell.Fr(maxRadius);
            for (int radius = 1; radius <= maxRadius; radius++) {
                List<Object> listEntry = Arrays.asList(cell.reproduced, (double) radius, frListCell.get(radius));
                frList.add(listEntry);
            }
        }

        Map<List<Double>, List<Integer>> frListBinned = new HashMap<>();
        for (List<Object> listEntry : frList) {
            int reproduced = (boolean) listEntry.get(0) ? 1 : 0;
            double radius = (double) listEntry.get(1);
            double fr = (double) listEntry.get(2);
            fr = Math.round(fr * Math.pow(10, 2)) / Math.pow(10, 2);
            List<Double> listBinnedKey = Arrays.asList(radius, fr);
            if (frListBinned.get(listBinnedKey) == null) {
                List<Integer> listBinnedEntry = Arrays.asList(reproduced, 1);
                frListBinned.put(listBinnedKey, listBinnedEntry);
            }
            else {
                List<Integer> listBinnedEntry = Arrays.asList(frListBinned.get(listBinnedKey).get(0)+reproduced, frListBinned.get(listBinnedKey).get(1)+1);
                frListBinned.put(listBinnedKey, listBinnedEntry);
            }
        }

        return frListBinned;
    }

    public static Map<Integer, Integer[]> GetPairCorrelation(Model2D model, int maxDistance, int area, Map<List<Integer>, Integer> annulusAreaLookupTable) {
        List<Cell2D> cells = new ArrayList<Cell2D>();
        for (Cell2D cell : model) {
            cells.add(cell);
        }

        Map<Integer, Integer[]> annulusRow = new HashMap<>();
        for (int i = 1; i < maxDistance; i++) {
            Integer annulusStartingList[] = {0, 0, 0, 0};
            annulusRow.put(i, annulusStartingList);
        }

        for (int a = 0; a < cells.size(); a++) {
            for (int b = 0; b < cells.size(); b++) {
                if (a == b) {
                    continue;
                }

                Cell2D cellA = cells.get(a);
                Cell2D cellB = cells.get(b);
                int cellAtype = cellA.type;
                int cellBtype = cellB.type;
                int xDistance = Math.abs(cellA.Xsq() - cellB.Xsq());
                int yDistance = Math.abs(cellA.Ysq() - cellB.Ysq());
                int annulus = xDistance + yDistance;

                if (annulus >= maxDistance) {
                    continue;
                }

                int pairType;
                if (cellAtype == 0 && cellBtype == 0)
                    pairType = 0;
                else if (cellAtype == 1 && cellBtype == 1)
                    pairType = 1;
                else if (cellAtype == 1 && cellBtype == 0)
                    pairType = 2;
                else if (cellAtype == 0 && cellBtype == 1)
                    pairType = 3;
                else
                    pairType = -9;

                List<Integer> tableKey = Arrays.asList(cellA.Xsq(), cellA.Ysq(), annulus);
                int normalized_count = (int)(area/annulusAreaLookupTable.get(tableKey));
                annulusRow.get(annulus)[pairType] = annulusRow.get(annulus)[pairType]+normalized_count;
            }
        }

        return annulusRow;
    }

    public static Map<List<Integer>, Integer> GetAnnulusAreaLookupTable(int x, int y, int maxDistance) {
        //TODO make more efficient (don't calculate area when radius is not near boundary, reflections)
        Map<List<Integer>, Integer> table = new HashMap<>();
        for (int r = 1; r < maxDistance; r++) {
            for (int x1 = 0; x1 < x; x1++) {
                for (int y1 = 0; y1 < y; y1++) {
                    List<Integer> tableKey = Arrays.asList(x1, y1, r);
                    int area = 0;
                    for (int x2 = Math.max(x1-r,0); x2 <= Math.min(x1+r,x); x2++) {
                        for (int y2 = Math.max(y1-r,0); y2 <= Math.min(y1+r,y); y2++) {
                            if (Math.abs(x1-x2) + Math.abs(y1-y2) == r) {
                                area += 1;
                            }
                        }
                    }
                    table.put(tableKey, area);
                }
            }
        }
        return table;
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
        int writePcFrequency = (int) params.get("writePcFrequency");
        int writeFsFrequency = (int) params.get("writeFsFrequency");
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
        boolean writeFs = writeFsFrequency != 0;
        boolean writePc = writePcFrequency != 0;
        boolean visualize = visualizationFrequency != 0;

        GridWindow win = null;
        GifMaker gifWin = null;
        if (visualize) {
            win = new GridWindow("SpatialEGT", x, y, 4);
            gifWin = new GifMaker(saveLoc+"growth.gif", 0, false);
            writePop = false;
            writeModel = false;
        }
        FileIO modelOut = null;
        if (writeModel) {
            modelOut = new FileIO(saveLoc+"coords.csv", "w");
            modelOut.Write("model,time,type,x,y\n");
        }
        FileIO popsOut = null;
        if (writePop) {
            popsOut = new FileIO(saveLoc+"populations.csv", "w");
            popsOut.Write("model,time,sensitive,resistant\n");
        }
        FileIO fsOut = null;
        FileIO frOut = null;
        if (writeFs) {
            fsOut = new FileIO(saveLoc+"fs.csv", "w");
            fsOut.Write("model,time,radius,fs,reproduced,total\n");
            frOut = new FileIO(saveLoc+"fr.csv", "w");
            frOut.Write("model,time,radius,fr,reproduced,total\n");
        }
        FileIO pcOut = null;
        int maxDistance = 0;
        Map<List<Integer>, Integer> annulusAreaLookupTable = null;
        if (writePc) {
            maxDistance = x;
            annulusAreaLookupTable = GetAnnulusAreaLookupTable(x, y, maxDistance);
            pcOut = new FileIO(saveLoc+"pairCorrelations.csv", "w");
            pcOut.Write("model,time,pair,measure,radius,normalized_count\n");
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
                        List<List<Integer>> coordLists = GetModelCoords(model);
                        List<Integer> cellTypes = coordLists.get(0);
                        List<Integer> xCoords = coordLists.get(1);
                        List<Integer> yCoords = coordLists.get(2);
                        for (int i = 0; i < coordLists.get(0).size(); i++) {
                            pcOut.Write(modelName+","+tick+","+cellTypes.get(i)+","+xCoords.get(i)+","+yCoords.get(i)+"\n");
                        }
                    }
                }
                if (writePc) {
                    if ((tick % writePcFrequency == 0) && (tick > 0)) {
                        Map<Integer, Integer[]> annulusRows = GetPairCorrelation(model, maxDistance, x*y, annulusAreaLookupTable);
                        String pairTypes[] = {"SS", "RR", "RS", "SR"};
                        for (int dist = 1; dist < maxDistance; dist++) {
                            for (int i = 0; i < 4; i++) {
                                pcOut.Write(modelName+","+tick+","+pairTypes[i]+",annulus,"+dist+","+annulusRows.get(dist)[i]+"\n");
                            }
                        }
                    }
                }
                if (writeFs) {
                    if ((tick % writeFsFrequency == 0) && (tick > 0)) {
                        Map<List<Double>, List<Integer>> fsList = GetFsList(model, 10);
                        for (Map.Entry<List<Double>,List<Integer>> entry : fsList.entrySet()) {
                        List<Double> key = entry.getKey();
                        List<Integer> value = entry.getValue();
                            fsOut.Write(modelName+","+tick+","+key.get(0)+","+key.get(1)+","+value.get(0)+","+value.get(1)+"\n");
                        }
                        Map<List<Double>, List<Integer>> frList = GetFrList(model, 10);
                        for (Map.Entry<List<Double>,List<Integer>> entry : frList.entrySet()) {
                        List<Double> key = entry.getKey();
                        List<Integer> value = entry.getValue();
                            frOut.Write(modelName+","+tick+","+key.get(0)+","+key.get(1)+","+value.get(0)+","+value.get(1)+"\n");
                        }
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
        if (writeModel) {
            modelOut.Close();
        }
        if (writePop) {
            popsOut.Close();
        }
        if (writePc) {
            pcOut.Close();
        }
        if (writeFs) {
            fsOut.Close();
            frOut.Close();
        }
    }
}
