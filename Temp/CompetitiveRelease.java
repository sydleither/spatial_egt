//https://halloworld.org/setup.html#example

package Temp;

import HAL.GridsAndAgents.AgentGrid2D;
import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.GridsAndAgents.PDEGrid2D;
import HAL.Gui.GridWindow;
import HAL.Tools.FileIO;
import HAL.Rand;
import HAL.Util;

class ReleaseModel extends AgentGrid2D<ReleaseCell> {
    public final int RESISTANT = Util.RGB(0,1,0), SENSITIVE = Util.RGB(0,0,1);
    public double DIV_PROB_SEN = 0.025, DIV_PROB_RES = 0.01, DEATH_PROB = 0.001,
                  DRUG_DIFF_RATE = 2, DRUG_UPTAKE = 0.91, DRUG_TOXICITY = 0.2,
                  DRUG_BOUNDARY_VAL = 1.0;
    public int DRUG_START = 400, DRUG_CYCLE = 200, DRUG_DURATION = 40;

    public PDEGrid2D drug;
    public Rand rng;
    public int[] divHood = Util.MooreHood(false);

    public ReleaseModel(int x, int y, Rand generator) {
        super(x, y, ReleaseCell.class);
        rng = generator;
        drug = new PDEGrid2D(x, y);
    }

    public void InitTumor(int radius, double resistanceProb) {
        int[] tumorNeighorhood = Util.CircleHood(true, radius);
        int hoodSize = MapHood(tumorNeighorhood, xDim/2, yDim/2);
        for (int i = 0; i < hoodSize; i++) {
            if (rng.Double() < resistanceProb) {
                NewAgentSQ(tumorNeighorhood[i]).type = RESISTANT;
            } else {
                NewAgentSQ(tumorNeighorhood[i]).type = SENSITIVE;
            }
        }
    }

    public void ModelStep(int tick) {
        ShuffleAgents(rng);
        for (ReleaseCell cell : this) {
            cell.CellStep();
        }
        int periodTick = (tick - DRUG_START) % DRUG_CYCLE;
        if (periodTick > 0 && periodTick < DRUG_DURATION) {
            drug.DiffusionADI(DRUG_DIFF_RATE, DRUG_BOUNDARY_VAL);
        } else {
            drug.DiffusionADI(DRUG_DIFF_RATE);
        }
        drug.Update();
    }

    public void DrawModel(GridWindow vis, int iModel) {
        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                ReleaseCell drawMe = GetAgent(x, y);
                if (drawMe != null) {
                    vis.SetPix(x+iModel*xDim, y, drawMe.type);
                } else {
                    vis.SetPix(x+iModel*xDim, y, Util.HeatMapRGB(drug.Get(x, y)));
                }
            }
        }
    }
}

class ReleaseCell extends AgentSQ2Dunstackable<ReleaseModel> {
    public int type;

    public void CellStep() {
        G.drug.Mul(Isq(), G.DRUG_UPTAKE);
        double deathProb, divProb;

        if (this.type == G.RESISTANT) {
            deathProb = G.DEATH_PROB;
        } else {
            deathProb = G.DEATH_PROB + G.drug.Get(Isq()) * G.DRUG_TOXICITY;
        }
        if (G.rng.Double() < deathProb) {
            Dispose();
            return;
        }

        if (this.type == G.RESISTANT) {
            divProb = G.DIV_PROB_RES;
        } else {
            divProb = G.DIV_PROB_SEN;
        }
        if (G.rng.Double() < divProb) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                G.NewAgentSQ(G.divHood[G.rng.Int(options)]).type = this.type;
            }
        }
    }
}

public class CompetitiveRelease {
    public static void main(String[] args) {
        int x = 100, y = 100, tumorRad = 10;
        int visScale = 3, msPause = 0;
        double resistanceProb = 0.5;
        GridWindow win = new GridWindow("Competitive Release", x*3, y, visScale);
        FileIO popsOut = new FileIO("populations.csv", "w");
        popsOut.Write("no_drug, constant_drug, pulsed_drug\n");

        ReleaseModel[] models = new ReleaseModel[3];
        for (int i = 0; i < models.length; i++) {
            models[i] = new ReleaseModel(x, y, new Rand());
            models[i].InitTumor(tumorRad, resistanceProb);
        }
        models[0].DRUG_DURATION = 0;
        models[1].DRUG_DURATION = 200;

        for (int tick = 0; tick <= 5000; tick++) {
            win.TickPause(msPause);
            for (int i = 0; i < models.length; i++) {
                models[i].ModelStep(tick);
                models[i].DrawModel(win, i);
            }
            popsOut.Write(models[0].Pop() + "," + models[1].Pop() + "," + models[2].Pop() + "\n");
            if (tick % 1000 == 0) {
                win.ToPNG("ModelsTick" + tick + ".png");
            }
        }

        popsOut.Close();
        win.Close();
    }
}
