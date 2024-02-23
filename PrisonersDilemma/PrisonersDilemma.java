//Inspired by (Coggan & Page, 2022)

package PrisonersDilemma;

import HAL.GridsAndAgents.AgentGrid2D;
import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class PrisonersDilemma {
    public static void main(String[] args) {
        //Initialize parameters
        int x = 100, y = 100;
        int tumorRadius = 10;
        int visScale = 3, msPause = 0;
        double deathProb = 0.2, divProb = 0.5;
        double defectorProportion = 0.1;
        int cost = 1, benefit = 2;
        GridWindow win = new GridWindow("Prisoner's Dilemma", x*2, y, visScale);
        
        Model2D model2d = new Model2D(x, y, new Rand(), cost, benefit, divProb, deathProb);
        model2d.InitTumor(tumorRadius, defectorProportion);

        Model0D model0d = new Model0D(x, y, new Rand(), cost, benefit, divProb, deathProb);
        model0d.InitTumor(0.1, defectorProportion);

        for (int tick = 0; tick <= 250; tick++) {
            win.TickPause(msPause);
            model2d.ModelStep(tick);
            model2d.DrawModel(win, 0);
            model0d.ModelStep(tick);
            model0d.DrawModel(win, 1);
            if (tick % 50 == 0) {
                win.ToPNG("ModelsTick" + tick + ".png");
            }
        }
        win.Close();
    }
}
