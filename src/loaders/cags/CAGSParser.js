import * as CAGS from 'cags.js';
import { UncompressedSplatArray } from '../UncompressedSplatArray.js';

// Attribute configuration based on example.ts
const ATTRIBUTES_LIST = {
    scaling: 6,
    rotation_re: 4,
    rotation_im: 6,
    opacity: 5,
    features_dc: 6,
    features_rest_0: 5,
    features_rest_1: 0,
    features_rest_2: 0,
};

export class CAGSParser {

    static getAttributesList() {
        return ATTRIBUTES_LIST;
    }

    static async parseToUncompressedSplatArray(basePath, sphericalHarmonicsDegree = 0) {
        const deQuantizer = new CAGS.DeQuantizer();

        try {
            // Load codebooks
            await CAGSParser.loadCodebooks(deQuantizer, basePath);

            // Load codes
            await CAGSParser.loadCodes(deQuantizer, basePath);

            // Dequantize the data
            const attributes = await deQuantizer.dequantize();

            // Convert to UncompressedSplatArray
            const splatArray = CAGSParser.convertDeQuantizedAttributesToSplatArray(attributes, sphericalHarmonicsDegree);

            return splatArray;

        } finally {
            // Clean up
            deQuantizer.dispose();
        }
    }

    static async loadCodebooks(deQuantizer, basePath) {
        // Load base layer codebook
        await deQuantizer.loadBaseLayerCodebook(`${basePath}/point_cloud.codebook.npz`);

        // Load enhancement layer codebooks
        for (const attr in ATTRIBUTES_LIST) {
            if (ATTRIBUTES_LIST.hasOwnProperty(attr)) {
                for (let i = 1; i <= ATTRIBUTES_LIST[attr]; i++) {
                    await deQuantizer.loadEnhancementLayerCodebook(attr, `${basePath}/point_cloud.layer.${attr}.${i}.codebook.npz`);
                }
            }
        }
    }

    static async loadCodes(deQuantizer, basePath) {
        // Load base layer codes
        await deQuantizer.loadBaseLayerCodes(`${basePath}/point_cloud.drc`);

        // Load enhancement layer codes
        for (const attr in ATTRIBUTES_LIST) {
            if (ATTRIBUTES_LIST.hasOwnProperty(attr)) {
                for (let i = 1; i <= ATTRIBUTES_LIST[attr]; i++) {
                    await deQuantizer.loadEnhancementLayerCodes(attr, `${basePath}/point_cloud.layer.${attr}.${i}.codes.npz`);
                }
            }
        }
    }

    static convertDeQuantizedAttributesToSplatArray(attributes, sphericalHarmonicsDegree = 0) {
        const splatArray = new UncompressedSplatArray(sphericalHarmonicsDegree);

        // Get position data from deQuantizer - the position data should be available directly
        let positions;
        let numSplats;

        // Handle different possible attribute structures
        if (attributes.position) {
            // Direct position array
            positions = attributes.position;
            numSplats = positions.length / 3;
        } else if (attributes.positions) {
            // Alternative naming
            positions = attributes.positions;
            numSplats = positions.length / 3;
        } else {
            // Position data might be in a different format, try to extract from the deQuantizer
            throw new Error('Invalid DeQuantizedAttributes: missing position data. Expected attributes.position or attributes.positions');
        }

        // Extract tensor data if needed
        const getDataArray = async (tensor) => {
            if (!tensor) return null;
            if (typeof tensor.data === 'function') {
                return await tensor.data();
            }
            if (Array.isArray(tensor) || tensor instanceof Float32Array || tensor instanceof Float64Array) {
                return tensor;
            }
            if (tensor.dataSync && typeof tensor.dataSync === 'function') {
                return tensor.dataSync();
            }
            return tensor;
        };

        // Process attributes synchronously if possible, or handle async data extraction
        const processAttributes = async () => {
            const scaleData = await getDataArray(attributes.scale || attributes.scaling);
            const rotationData = await getDataArray(attributes.rotation);
            const opacityData = await getDataArray(attributes.opacity);
            const featuresData = await getDataArray(attributes.features_dc);

            for (let i = 0; i < numSplats; i++) {
                const baseIndex = i * 3;

                // Extract position
                const x = positions[baseIndex];
                const y = positions[baseIndex + 1];
                const z = positions[baseIndex + 2];

                // Extract scaling (default to 1 if not available)
                let scaleX = 1;
                let scaleY = 1;
                let scaleZ = 1;
                if (scaleData && scaleData.length >= (i + 1) * 3) {
                    scaleX = Math.exp(scaleData[baseIndex]);
                    scaleY = Math.exp(scaleData[baseIndex + 1]);
                    scaleZ = Math.exp(scaleData[baseIndex + 2]);
                }

                // Extract rotation (quaternion - default to identity if not available)
                let rotX = 0;
                let rotY = 0;
                let rotZ = 0;
                let rotW = 1;
                if (rotationData) {
                    const rotIndex = i * 4;
                    if (rotationData.length >= (i + 1) * 4) {
                        rotW = rotationData[rotIndex]; // w component first
                        rotX = rotationData[rotIndex + 1]; // x component
                        rotY = rotationData[rotIndex + 2]; // y component
                        rotZ = rotationData[rotIndex + 3]; // z component

                        // Normalize quaternion
                        const norm = Math.sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ + rotW * rotW);
                        if (norm > 0) {
                            rotX /= norm;
                            rotY /= norm;
                            rotZ /= norm;
                            rotW /= norm;
                        }
                    }
                }

                // Extract color (features_dc - default to white if not available)
                let r = 255;
                let g = 255;
                let b = 255;
                if (featuresData && featuresData.length >= (i + 1) * 3) {
                    const SH_C0 = 0.28209479177387814;
                    r = Math.max(0, Math.min(255, Math.floor((0.5 + SH_C0 * featuresData[baseIndex]) * 255)));
                    g = Math.max(0, Math.min(255, Math.floor((0.5 + SH_C0 * featuresData[baseIndex + 1]) * 255)));
                    b = Math.max(0, Math.min(255, Math.floor((0.5 + SH_C0 * featuresData[baseIndex + 2]) * 255)));
                }

                // Extract opacity (default to 255 if not available)
                let opacity = 255;
                if (opacityData && opacityData.length > i) {
                    // Convert from logit space to [0, 255]
                    const logitOpacity = opacityData[i];
                    opacity = Math.max(0, Math.min(255, Math.floor((1 / (1 + Math.exp(-logitOpacity))) * 255)));
                }

                // Add splat to array
                splatArray.addSplatFromComonents(
                    x, y, z,
                    scaleX, scaleY, scaleZ,
                    rotX, rotY, rotZ, rotW,
                    r, g, b, opacity
                );
            }

            return splatArray;
        };
        return processAttributes();
    }
}
