package SpatialEGT;

import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Util;

public class Cell2D extends AgentSQ2Dunstackable<Model2D> {
    int type;
    int color;
    double deathRate, divRate;

    public void Init(int type) {
        this.type = type;
        this.deathRate = G.deathRate;
        if (type == 0) {
            this.color = Util.RGB256(76, 149, 108);
            this.divRate = G.divRateS;
        }
        else {
            this.color = Util.RGB256(239, 124, 142);
            this.divRate = G.divRateR;
        }
    }

    public void CellStep() {
        //divison + drug death
        if (G.rng.Double() < this.divRate) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                if (G.rng.Double() < G.drugKillRate * G.drugConcentration * (1-this.type)) {
                    Dispose();
                    return;
                }
                else {
                    G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init(this.type);
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