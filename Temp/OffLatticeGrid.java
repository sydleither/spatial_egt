//https://www.youtube.com/watch?v=lGXVRbSBFLg

package Temp;

import HAL.GridsAndAgents.AgentPT2D;
import HAL.GridsAndAgents.AgentGrid2D;
import HAL.Gui.GridWindow;
import HAL.Gui.OpenGL2DWindow;
import HAL.Rand;
import HAL.Util;

class ExamplePTCell extends AgentPT2D<OffLatticeGrid>{
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
        if(G.PopAt(Isq()) < 5 && G.rng.Double() < divProb) {
            G.NewAgentPT(Xpt(), Ypt()).Init();
        }
        //cell movement
        G.rng.RandomPointInCircle(0.5, G.moveCoords);
        MoveSafePT(Xpt()+G.moveCoords[0], Ypt()+G.moveCoords[1]);
    }
}

public class OffLatticeGrid extends AgentGrid2D<ExamplePTCell>{
    Rand rng = new Rand();
    double[] moveCoords = new double[2];

    public OffLatticeGrid(int x, int y) {
        super(x, y, ExamplePTCell.class);
    }
    public void StepCells(double dieProb, double divProb){
        for(ExamplePTCell cell : this){
            cell.StepCell(dieProb, divProb);
        }
    }
    public void DrawModel(OpenGL2DWindow win){
        win.Clear(Util.BLACK);
        for (ExamplePTCell cell : this){
            win.Circle(cell.Xpt(), cell.Ypt(), 0.5, cell.color);
        }
        win.Update();
    }
    public static void main(String[] args){
        int x = 100;
        int y = 100;
        int timesteps = 1000;
        double dieProb = 0.1;
        double divProb = 0.3;

        OpenGL2DWindow win = new OpenGL2DWindow("2D", 500, 500, x, y);
        OffLatticeGrid model = new OffLatticeGrid(x, y);

        model.NewAgentSQ(model.xDim/2, model.yDim/2).Init();

        for (int i = 0; i < timesteps; i ++){
            if(win.IsClosed()){
                break;
            }
            win.TickPause(100);

            if(model.Pop() == 0){
                model.NewAgentSQ(model.xDim/2, model.yDim/2).Init();
            }

            model.StepCells(dieProb, divProb);
            model.DrawModel(win);
        }
        win.Close();
    }
}