package SpatialEGT;

import java.util.HashMap;

import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Util;

public class Cell2D extends AgentSQ2Dunstackable<Model2D> {
    int type;
    int color;
    double deathRate;
    boolean reproduced;

    public void Init(int type) {
        this.type = type;
        this.deathRate = G.deathRate;
        if (type == 0) {
            this.color = Util.RGB256(76, 149, 108);
        }
        else {
            this.color = Util.RGB256(239, 124, 142);
        }
    }

    public double GetDivRate() {
        double total_payoff = 0;
        int neighbors = MapOccupiedHood(G.gameHood);
        if (neighbors == 0) {
            return G.payoff[this.type][this.type];
        }
        for (int i = 0; i < neighbors; i++) {
            Cell2D neighborCell = G.GetAgent(G.gameHood[i]);
            total_payoff += G.payoff[this.type][neighborCell.type];
        }
        return total_payoff/neighbors;
    }

    public void CellStep() {
        //divison + drug effects
        double divRate = this.GetDivRate();
        if (G.drugConcentration > 0.0 && this.type == 0) {
            divRate = divRate * (1 - G.drugGrowthReduction);
        }
        reproduced = false;
        if (G.rng.Double() < divRate) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init(this.type);
            }
            reproduced = true;
        }
        //natural death
        if (G.rng.Double() < G.deathRate) {
            Dispose();
            return;
        }
    }

    public HashMap<Integer,Double> Fs(int maxRadius) {
        HashMap<Integer,Double> fsList = new HashMap<Integer,Double>();
        for (int radius = 1; radius <= maxRadius; radius++) {
            int[] neighborhood = Util.CircleHood(false, radius);
            int neighbors = MapOccupiedHood(neighborhood);
            if (neighbors == 0) {
                fsList.put(radius, 0.0);
                continue;
            }
            double s = 0;
            double r = 0;
            for (int i = 0; i < neighbors; i++) {
                Cell2D neighborCell = G.GetAgent(neighborhood[i]);
                if (neighborCell.type == 0)
                    s += 1;
                else
                    r += 1;
            }
            fsList.put(radius, s/(s+r));
        }
        return fsList;
    }

    public HashMap<Integer,Double> Fr(int maxRadius) {
        HashMap<Integer,Double> frList = new HashMap<Integer,Double>();
        for (int radius = 1; radius <= maxRadius; radius++) {
            int[] neighborhood = Util.CircleHood(false, radius);
            int neighbors = MapOccupiedHood(neighborhood);
            if (neighbors == 0) {
                frList.put(radius, 0.0);
                continue;
            }
            double s = 0;
            double r = 0;
            for (int i = 0; i < neighbors; i++) {
                Cell2D neighborCell = G.GetAgent(neighborhood[i]);
                if (neighborCell.type == 0)
                    s += 1;
                else
                    r += 1;
            }
            frList.put(radius, r/(s+r));
        }
        return frList;
    }
}