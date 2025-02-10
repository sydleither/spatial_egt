package SpatialEGT;

import HAL.GridsAndAgents.AgentSQ3Dunstackable;
import HAL.Util;

public class Cell3D extends AgentSQ3Dunstackable<Model3D> {
    int type;
    double deathRate;

    public void Init(int type) {
        this.type = type;
        this.deathRate = G.deathRate;
    }

    public double GetDivRate() {
        double total_payoff = 0;
        int neighbors = MapOccupiedHood(G.gameHood);
        if (neighbors == 0) {
            return G.payoff[this.type][this.type];
        }
        for (int i = 0; i < neighbors; i++) {
            Cell3D neighborCell = G.GetAgent(G.gameHood[i]);
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
        if (G.rng.Double() < divRate) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init(this.type);
            }
        }
        //natural death
        if (G.rng.Double() < G.deathRate) {
            Dispose();
            return;
        }
    }
}