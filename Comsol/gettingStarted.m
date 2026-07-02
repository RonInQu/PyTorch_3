import com.comsol.model.*
import com.comsol.model.util.*
model = ModelUtil.create('Test');
comp1 = model.component.create('comp1', true);
geom1 = comp1.geom.create('geom1', 3);
geom1.lengthUnit('mm');
imp1 = geom1.create('imp1', 'Import');
imp1.set('filename', fullfile(pwd, 'G25.STEP'));
imp1.set('type', 'cad');
imp1.importData();
geom1.run('fin');
mphgeom(model, 'geom1', 'FaceLabels', 'on');
mphsave(model, fullfile(pwd, 'geometry_check.mph'));