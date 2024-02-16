//https://www.youtube.com/watch?v=TMnB7nNLrV4

package Temp;

import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.GridsAndAgents.AgentGrid2D;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

class ExampleCell extends AgentSQ2Dunstackable<OnLatticeGrid>{
    int color;

    public void Init(){
        this.color = Util.RGB(G.rng.Double(), G.rng.Double(), G.rng.Double());
    }

    public void StepCell(double dieProb, double divProb) {
        //cell death
        if(G.rng.Double() < dieProb) {
            Dispose();
            return;
        }
        //cell division
        if(G.rng.Double() < divProb) {
            int options = MapEmptyHood(G.divHood);
            if(options > 0){
                G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init();
            }
        }
    }
}

public class OnLatticeGrid extends AgentGrid2D<ExampleCell>{
    Rand rng = new Rand();
    int[] divHood = Util.VonNeumannHood(false);

    public OnLatticeGrid(int x, int y) {
        super(x, y, ExampleCell.class);
    }
    public void StepCells(double dieProb, double divProb){
        for(ExampleCell cell : this){
            cell.StepCell(dieProb, divProb);
        }
    }
    public void DrawModel(GridWindow win){
        for(int i = 0; i < length; i++){
            int color = Util.BLACK;
            ExampleCell cell = GetAgent(i);
            if(cell != null){
                color = cell.color;
            }
            win.SetPix(i, color);
        }
    }
    public static void main(String[] args){
        int x = 100;
        int y = 100;
        int timesteps = 1000;
        double dieProb = 0.1;
        double divProb = 0.3;

        GridWindow win = new GridWindow(x, y, 4);
        OnLatticeGrid model = new OnLatticeGrid(x, y);

        model.NewAgentSQ(model.xDim/2, model.yDim/2).Init();

        for (int i = 0; i < timesteps; i ++){
            win.TickPause(100);

            if(model.Pop() == 0){
                model.NewAgentSQ(model.xDim/2, model.yDim/2).Init();
            }

            model.StepCells(dieProb, divProb);
            model.DrawModel(win);
        }
    }
}