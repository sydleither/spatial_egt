package SpatialEGT;

import HAL.GridsAndAgents.AgentSQ3Dunstackable;
import HAL.Util;

public class Cell3D extends AgentSQ3Dunstackable<Model3D> {
    int type;
    int color;
    double deathRate;
    boolean interacting;

    public void Init(int type, boolean interacting) {
        this.type = type;
        this.interacting = interacting;
        this.deathRate = G.deathRate;
        if (type == 0) {
            this.color = Util.RGB256(76, 149, 108);
        }
        else {
            this.color = Util.RGB256(239, 124, 142);
        }
    }

    public double GetDivRate() {
        if (!this.interacting) {
            return type == 0 ? G.divRateS : G.divRateR;
        }
        else {
            int total_payoff = 0;
            int neighbors = MapOccupiedHood(G.divHood);
            if (neighbors == 0) {
                return 0.005;
            }
            else if (neighbors == 4) {
                return 0.0;
            }
            for (int i = 0; i < neighbors; i++) {
                Cell3D neighborCell = G.GetAgent(G.divHood[i]);
                total_payoff += G.payoff[this.type][neighborCell.type];
            }
            return total_payoff/100.0;
        }
    }

    public void CellStep() {
        //divison + drug death
        double divRate = this.GetDivRate();
        if (G.rng.Double() < divRate) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                if (G.rng.Double() < G.drugKillRate * G.drugConcentration * (1-this.type)) {
                    Dispose();
                    return;
                }
                else {
                    G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init(this.type, this.interacting);
                }
            }
        }

        //natural death
        if (G.rng.Double() < G.deathRate) {
            Dispose();
            return;
        }
    }
}