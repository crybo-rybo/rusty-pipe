use pyo3::prelude::*;

/// Represents a single object detection.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Detection {
    #[pyo3(get, set)]
    pub class_id: i32,
    #[pyo3(get, set)]
    pub conf: f32,
    #[pyo3(get, set)]
    pub bbox: (f32, f32, f32, f32), // x, y, width, height
}

#[pymethods]
impl Detection {
    #[new]
    fn new(class_id: i32, conf: f32, bbox: (f32, f32, f32, f32)) -> Self {
        Detection { class_id, conf, bbox }
    }
    
    fn __repr__(&self) -> String {
        format!("Detection(id={}, conf={:.2}, bbox={:?})", self.class_id, self.conf, self.bbox)
    }
}

/// Processes a list of detections.
/// Currently filters for "Person" class (ID 0) with confidence > 0.5.
#[pyfunction]
fn process_frame(detections: Vec<Detection>) -> PyResult<Vec<Detection>> {
    let filtered: Vec<Detection> = detections
        .into_iter()
        .filter(|d| d.class_id == 0 && d.conf > 0.5) // 0 is usually 'person' in COCO
        .collect();
    
    Ok(filtered)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_cv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detection>()?;
    m.add_function(wrap_pyfunction!(process_frame, m)?)?;
    Ok(())
}
