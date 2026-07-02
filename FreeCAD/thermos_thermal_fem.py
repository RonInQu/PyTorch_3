import FreeCAD as App
import Part
import Fem
import ObjectsFem

try:
    App.closeDocument("Thermos_Transient_Thermal")
except:
    pass

doc = App.newDocument("Thermos_Transient_Thermal")

# ==========================================
# 1. GEOMETRY 
# ==========================================
r_inner = 40.0
r_gap = 42.0
r_outer = 45.0
height = 200.0
floor_thickness = 5.0  

inner_inside = Part.makeCylinder(r_inner, height)
inner_inside.translate(App.Vector(0, 0, floor_thickness)) 
inner_outside = Part.makeCylinder(r_gap, height + floor_thickness)
inner_wall_shape = inner_outside.cut(inner_inside)

inner_wall_obj = doc.addObject("Part::Feature", "Inner_Wall")
inner_wall_obj.Shape = inner_wall_shape

vacuum_outside = Part.makeCylinder(r_outer, height + (floor_thickness * 2))
vacuum_shape = vacuum_outside.cut(inner_outside)

vacuum_obj = doc.addObject("Part::Feature", "Vacuum_Gap")
vacuum_obj.Shape = vacuum_shape

thermos_compound = doc.addObject("Part::Compound", "Thermos_Bottle")
thermos_compound.Links = [inner_wall_obj, vacuum_obj]
doc.recompute()

# ==========================================
# 2. FEM ANALYSIS SETUP
# ==========================================
analysis = ObjectsFem.makeAnalysis(doc, "Analysis")
solver = ObjectsFem.makeSolverElmer(doc, "SolverElmer")
analysis.addObject(solver)

solver.SimulationType = "Transient"
solver.TimestepIntervals = [120]  
solver.TimestepSizes = [60]      

heat_eq = ObjectsFem.makeEquationHeat(doc, solver)
analysis.addObject(heat_eq)

# ==========================================
# 3. MATERIAL DEFINITIONS
# ==========================================
material_steel = ObjectsFem.makeMaterialSolid(doc, "Stainless_Steel")
mat_s = material_steel.Material
mat_s["Name"] = "Steel-Generic"
mat_s["ThermalConductivity"] = "16.0"  
mat_s["SpecificHeat"] = "500.0"       
mat_s["Density"] = "7800.0"           
analysis.addObject(material_steel)
# FIX: Use explicit object reference lists directly for Elmer compliance
material_steel.References = [inner_wall_obj]

material_vacuum = ObjectsFem.makeMaterialSolid(doc, "Pseudo_Vacuum")
mat_v = material_vacuum.Material
mat_v["Name"] = "Pseudo-Vacuum"
mat_v["ThermalConductivity"] = "0.0005"  
mat_v["SpecificHeat"] = "1.0"
mat_v["Density"] = "0.001"
analysis.addObject(material_vacuum)
material_vacuum.References = [vacuum_obj]

# ==========================================
# 4. BOUNDARY CONDITIONS
# ==========================================
initial_temp = ObjectsFem.makeConstraintInitialTemperature(doc, "Initial_Coffee_Temp")
valid_props = ["Temperature", "InitialTemperature", "Initial Temperature"]
found_prop = [p for p in initial_temp.PropertiesList if p in valid_props]
if found_prop:
    setattr(initial_temp, found_prop[0], 90.0)

# FIX: Map boundary directly to geometry object references
initial_temp.References = [inner_wall_obj]   
analysis.addObject(initial_temp)

convection = ObjectsFem.makeConstraintHeatflux(doc, "Outer_Air_Convection")
convection.ConstraintType = "Convection"  
convection.AmbientTemp = 20.0             
convection.FilmCoef = 10.0                
convection.References = [vacuum_obj]     
analysis.addObject(convection)

# ==========================================
# 5. GMSH MESH GENERATION
# ==========================================
fem_mesh = ObjectsFem.makeMeshGmsh(doc, "FEM_Mesh")
fem_mesh.Shape = thermos_compound
fem_mesh.CharacteristicLengthMax = 12.0  
analysis.addObject(fem_mesh)

doc.recompute()

if App.GuiUp:
    inner_wall_obj.ViewObject.hide()
    vacuum_obj.ViewObject.hide()
    thermos_compound.ViewObject.hide()
    fem_mesh.ViewObject.show()

print("Reference arrays formatted directly for Elmer exporter engine.")