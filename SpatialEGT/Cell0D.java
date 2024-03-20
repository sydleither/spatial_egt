package SpatialEGT;

import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Util;

public class Cell0D extends AgentSQ2Dunstackable<Model0D> {
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
            for (int i = 0; i < 5; i++) {
                int randCell = G.rng.Int(G.xDim*G.yDim);
                Cell0D neighborCell = G.GetAgent(randCell);
                if (neighborCell != null) {
                    total_payoff += G.payoff[this.type][neighborCell.type];
                }
            }
            if (total_payoff < 0.005) {
                return 0.005;
            }
            return total_payoff/125.0;
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